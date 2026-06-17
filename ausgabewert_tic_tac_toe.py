"""
Ausgabewert-Auswertung: Tic-Tac-Toe.

Berechnet Feature-Beiträge (Shapley, Banzhaf, …) für ein festes Brett:
9 Features = die 9 Felder. Gegner spielt MinMax.

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

    # Schritt 1–2: optimale (ε-greedy) Policy lernen
    train(agent, env, TRAIN_STEPS)
    agent.get_policy(env.valid_dict)          # π(a|s) nur für gültige Züge
    agent.get_value_table(env.valid_dict)     # V(s) = max_a Q(s,a)

    # Schritt 3: wie oft besucht der Agent welche Brettstellungen?
    state_dist = get_state_dist(agent, env, STATE_SAMPLES)

    # Schritt 4: π_C — Policy, wenn nur Koalition C der Features beobachtet wird
    F = np.arange(env.state_dim)
    ch = Characteristics(env, states)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states, env.valid_dict))
        for C in tqdm_label(F_not_i(F), "ttt pi_C")
    }
    # v_C — erwarteter Wert unter partieller Feature-Beobachtung
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), "ttt v_C")
    }

    # Schritt 5: Charakteristik v_C^π(s) für jede Koalition C (Monte-Carlo / Q-Werte)
    char_data = ch.fast_local_sverl_C_values(
        ch.pi_Cs, valid_dict=env.valid_dict, multi_process=False, num_p=1,
    )

    # Schritt 6: kooperative Aufteilungsregeln → Beitrag pro Feld (Feature)
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
    """
    Schreibt ``AUSGABEWERTE.md``: Brettdarstellung, Gesamt- und Detailtabelle.

    Nutzt ``mean_comparison_table`` (Mittel/Mapping pro Feld) und
    ``comparison_table`` (alle Methoden pro Feld für den erklärten Zustand).
    """
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
    """
    Ausführung: Ordner anlegen → Ausgabewerte berechnen → speichern und visualisieren.

    Erzeugt:
      - ``results.pkl``  — Rohdaten aller Methoden
      - ``vergleich_gesamt.png`` — Balken- und Heatmap-Vergleich der 9 Felder
      - ``AUSGABEWERTE.md`` — Textbericht mit Tabellen
    """
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
