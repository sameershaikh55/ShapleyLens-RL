"""
Ausgabewert-Auswertung: Tic-Tac-Toe.

Berechnet Feature-Beiträge (Shapley, Banzhaf, …) für ein festes Brett:
9 Features = die 9 Felder. Gegner spielt MinMax.

Ergebnisse → ``Ausgabewert/tic_tac_toe/``

    python ausgabewert_tic_tac_toe.py
    python ausgabewert_tic_tac_toe.py --plot-only
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from characteristics import Characteristics
from q_agent_2 import Agent
from ausgabewert_grid_worlds import (
    CHAR,
    METHODS,
    comparison_table,
    format_state,
    mean_comparison_table,
    save_comparison_plots,
)
from shapley import Shapley
from banzhaf import Banzhaf
from nucleolus import Nucleolus
from tau import TauValue
from utopia_payoff import UtopiaPayoff
from gately import Gately
from tic_tac_toe.tic_tac_toe import TTT
from utils import F_not_i, get_state_dist, train, tqdm_label

OUT = ROOT / "Ausgabewert" / "tic_tac_toe"
FEATURE_NAMES = [f"({r},{c})" for r in range(3) for c in range(3)]  # 9 Brettfelder
TRAIN_STEPS = 100_000       # Q-Learning-Schritte bis zur Policy
STATE_SAMPLES = 50_000      # Stichproben für die Zustandsverteilung p^π(s)

# Erklärter Zustand: Agent (1) am Zug, Gegner-Steine auf (2,0) und (2,2)
DEFAULT_STATE = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [2, 0, 2],
], dtype=int).flatten()


def run_tic_tac_toe() -> dict:
    """
    Hauptpipeline: Agent trainieren → Charakteristikwerte → Ausgabewerte.

    Schritte:
      1. Q-Learning-Agent gegen MinMax-Gegner trainieren
      2. Policy π(a|s) und Wertfunktion V(s) aus Q-Tabelle ableiten
      3. Stationäre Zustandsverteilung p^π(s) schätzen
      4. Für jede Feature-Koalition C: partielle Policies π_C und Werte v_C
      5. Local SVERL (fast): erwarteter Return pro Koalition und Zustand
      6. Shapley, Banzhaf, Nucleolus, Tau, Utopia, Gately auf Charakteristik anwenden

    Returns:
        dict mit ``results`` (Ausgabewerte pro Methode) und ``states`` (erklärte Zustände)
    """
    print(f"\n{'='*60}\n  TIC-TAC-TOE\n{'='*60}")

    env = TTT()
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=1, alpha=0.2)
    states = np.array([DEFAULT_STATE])

    train(agent, env, TRAIN_STEPS)
    agent.get_policy(env.valid_dict)
    agent.get_value_table(env.valid_dict)

    state_dist = get_state_dist(agent, env, STATE_SAMPLES)

    F = np.arange(env.state_dim)
    ch = Characteristics(env, states)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states, env.valid_dict))
        for C in tqdm_label(F_not_i(F), "ttt pi_C")
    }
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), "ttt v_C")
    }

    char_data = ch.fast_local_sverl_C_values(
        ch.pi_Cs, valid_dict=env.valid_dict, multi_process=False, num_p=1,
    )

    calculators = {
        "shapley": Shapley(states),
        "banzhaf": Banzhaf(states, normalized=True),
        "nucleolus": Nucleolus(states),
        "tau": TauValue(states, strict=False),
        "utopia": UtopiaPayoff(states, normalized=True),
        "gately": Gately(states),
    }
    results = {}
    for m in METHODS:
        try:
            results[m] = {CHAR: calculators[m].run(char_data)}
        except Exception as exc:
            print(f"  WARN {m}: {exc}")

    print(f"  Zustand: {format_state(DEFAULT_STATE)}")
    return {"results": results, "states": states}


def write_report(data: dict) -> None:
    """Schreibt ``AUSGABEWERTE.md`` mit Brett, Tabellen und Bildverweis."""
    results, states = data["results"], data["states"]
    board = DEFAULT_STATE.reshape(3, 3)
    body = [
        "# TIC-TAC-TOE — Ausgabewert-Vergleich\n\n",
        "Tic-Tac-Toe vs. MinMax. **9 Features** = Felder (0,0) … (2,2).\n\n",
        f"Charakteristik: **{CHAR}** | Train: {TRAIN_STEPS:,}\n\n",
        "**Erklärter Zustand (Brett):**\n\n",
        "```\n",
        "\n".join(" ".join(str(board[r, c]) for c in range(3)) for r in range(3)),
        "\n```\n",
        "(0=leer, 1=Agent, 2=Gegner)\n\n",
        "## Gesamtvergleich (pro Feld)\n\n",
        mean_comparison_table(results, states, FEATURE_NAMES),
        "\n\n![ttt](vergleich_gesamt.png)\n\n",
        "## Detailtabelle\n\n",
        comparison_table(results, states, FEATURE_NAMES),
        "\n",
    ]
    (OUT / "AUSGABEWERTE.md").write_text("".join(body), encoding="utf-8")
    print(f"Bericht: {OUT / 'AUSGABEWERTE.md'}")


def _load_or_run(plot_only: bool) -> dict:
    pkl_path = OUT / "results.pkl"
    states_path = OUT / "states.pkl"

    if plot_only:
        if not pkl_path.exists():
            raise FileNotFoundError(f"Keine Ergebnisse: {pkl_path}")
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
        if states_path.exists():
            with open(states_path, "rb") as f:
                states = pickle.load(f)
        else:
            states = np.array([DEFAULT_STATE])
        return {"results": results, "states": states}

    data = run_tic_tac_toe()
    with open(pkl_path, "wb") as f:
        pickle.dump(data["results"], f)
    with open(states_path, "wb") as f:
        pickle.dump(data["states"], f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Ausgabewert")
    parser.add_argument("--plot-only", action="store_true", help="Nur Plots aus results.pkl")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = _load_or_run(args.plot_only)

    save_comparison_plots(
        data["results"], data["states"], OUT, "tic_tac_toe", FEATURE_NAMES,
    )
    write_report(data)
    print(f"Fertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()
