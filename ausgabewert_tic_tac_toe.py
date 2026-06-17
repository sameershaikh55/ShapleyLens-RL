"""
Ausgabewert-Auswertung: Tic-Tac-Toe.

Ergebnisse → ``Ausgabewert/tic_tac_toe/``

    python ausgabewert_tic_tac_toe.py
"""

from __future__ import annotations

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
    NUM_ROLLS,
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
FEATURE_NAMES = [f"({r},{c})" for r in range(3) for c in range(3)]
TRAIN_STEPS = 100_000
STATE_SAMPLES = 50_000

# Standard-Brett aus tic_tac_toe/run.py (Agent am Zug, Gegner MinMax)
DEFAULT_STATE = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [2, 0, 2],
], dtype=int).flatten()


def run_tic_tac_toe() -> dict:
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
    results, states = data["results"], data["states"]
    board = DEFAULT_STATE.reshape(3, 3)
    body = [
        "# TIC-TAC-TOE — Ausgabewert-Vergleich\n\n",
        "Tic-Tac-Toe vs. MinMax. **9 Features** = Felder (0,0) … (2,2).\n\n",
        f"Charakteristik: **{CHAR}** | Train: {TRAIN_STEPS:,} | Rolls: {NUM_ROLLS:,}\n\n",
        "**Erklärter Zustand (Brett):**\n\n",
        "```\n",
        "\n".join(" ".join(str(board[r, c]) for c in range(3)) for r in range(3)),
        "\n```\n",
        "(0=leer, 1=Agent, 2=Gegner)\n\n",
        "## Gesamtvergleich (pro Feld)\n\n",
        mean_comparison_table(results, states, FEATURE_NAMES),
        "\n\n## Detailtabelle\n\n",
        comparison_table(results, states, FEATURE_NAMES),
        "\n",
    ]
    (OUT / "AUSGABEWERTE.md").write_text(
        "".join(body[:10]) + "\n\n![ttt](vergleich_gesamt.png)\n\n" + "".join(body[10:]),
        encoding="utf-8",
    )
    print(f"Bericht: {OUT / 'AUSGABEWERTE.md'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = run_tic_tac_toe()

    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump(data["results"], f)

    save_comparison_plots(
        data["results"], data["states"], OUT, "tic_tac_toe", FEATURE_NAMES,
    )
    write_report(data)
    print(f"Fertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()
