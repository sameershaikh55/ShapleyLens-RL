"""
Ausgabewert-Auswertung: Minesweeper.

Berechnet Feature-Beiträge (Shapley, Banzhaf, …) für ein 4×4-Minesweeper-Brett
mit 2 Minen. 16 Features = die 16 Brettfelder. Q-Learning-Training, danach
**fast_local_sverl** als Charakteristik (statt local_sverl wie bei den
anderen Spielen).

Abweichung vom sonstigen Auswertung-Muster: lokales SVERL braucht eine
Monte-Carlo-Rollout-Episode pro Koalition C. Bei 16 Features gibt es
2^15 = 32.768 Koalitionen (vs. z. B. 16 bei Taxi mit 4 Features) — das wäre
mit local_sverl praktisch nicht in vertretbarer Zeit berechenbar. fast_local_sverl
rollt die Episoden einmal pro Zustand/Aktion aus und wiederverwendet sie für
alle Koalitionen; das ist auch der Ansatz im main-Branch (minesweeper/run.py).

Nutzt ``q_agent_2`` (statt ``q_agent_1`` wie bei den Grid Worlds), weil in
Minesweeper nur unaufgedeckte Felder gültige Aktionen sind — die Policy muss
also pro Zustand wissen, welche Aktionen erlaubt sind (``env.valid_dict``).
Aus demselben Grund nutzte der Kollege für Tic-Tac-Toe ebenfalls ``q_agent_2``.

Ergebnisse → ``Ausgabewert/minesweeper/``

    python ausgabewert_minesweeper.py
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from characteristics import Characteristics
from q_agent_2 import Agent
from ausgabewert_grid_worlds import (
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
from minesweeper.minesweeper import Minesweeper
from utils import F_not_i, get_state_dist, train, find_states_minesweeper, tqdm_label

OUT = ROOT / "Ausgabewert" / "minesweeper"
BOARD_HEIGHT = 4
BOARD_LENGTH = 4
NUM_MINES = 2
FEATURE_NAMES = [f"({r},{c})" for r in range(BOARD_HEIGHT) for c in range(BOARD_LENGTH)]  # 16 Felder

# Eigene Charakteristik (statt der geteilten CHAR="local_sverl" aus
# ausgabewert_grid_worlds) — siehe Modul-Docstring oben für die Begründung.
MS_CHAR = "fast_local_sverl"
MS_NUM_ROLLS = 1_000  # Rollouts pro (Zustand, Aktion) — einmalig, nicht pro Koalition

TRAIN_STEPS = 3_000_000     # Q-Learning-Schritte (Minesweeper braucht deutlich mehr als die Grid Worlds)
FIND_STATES_STEPS = 1_000_000  # Schritte, um mögliche Mine-Konfigurationen für die Zielzustände zu sammeln
STATE_SAMPLES = 1_000_000   # Stichproben für die Zustandsverteilung p^π(s)

# Zwei Zustände mit identischem aufgedecktem Brett, aber unterschiedlicher
# Mine an Feld (3,2) bzw. (2,3) — zeigt, wie stark einzelne Felder den
# Ausgabewert je nach (für den Agenten unsichtbarer) Minenlage verschieben.
#   -1 = unaufgedeckt, 0..8 = Zahl, 2 = Mine bereits aufgedeckt
STATES_TO_EXPLAIN = np.array([
    [0,  0,  1, -1,
     0,  1,  2, -1,
     0,  1, -1, -1,
     0,  1,  1,  1],
    [0,  0,  1, -1,
     0,  1,  2, -1,
     0,  1, -1,  2,
     0,  1,  1,  1],
])

STATE_LABELS = {
    tuple(STATES_TO_EXPLAIN[0]): "Feld (3,2) noch unaufgedeckt",
    tuple(STATES_TO_EXPLAIN[1]): "Feld (3,2) als Mine aufgedeckt (Spiel verloren)",
}


def run_minesweeper() -> dict:
    """
    Hauptpipeline: Agent trainieren → Charakteristikwerte → Ausgabewerte.

    Schritte:
      1. Q-Learning-Agent trainieren (state-abhängige gültige Aktionen)
      2. Mögliche Mine-Konfigurationen für die beiden Zielzustände einsammeln
         (ein Zustand kann mehrere Minenlagen haben — Minesweeper ist stochastisch)
      3. Policy π(a|s) und Wertfunktion V(s) aus Q-Tabelle ableiten
      4. Stationäre Zustandsverteilung p^π(s) schätzen
      5. Für jede Feature-Koalition C: partielle Policies π_C und Werte v_C
      6. Local SVERL (Monte-Carlo): erwarteter Return pro Koalition und Zustand
      7. Shapley, Banzhaf, Nucleolus, Tau, Utopia, Gately anwenden

    Returns:
        dict mit ``results`` (Ausgabewerte pro Methode) und ``states`` (erklärte Zustände)
    """
    print(f"\n{'='*60}\n  MINESWEEPER\n{'='*60}")

    env = Minesweeper(length=BOARD_LENGTH, height=BOARD_HEIGHT, num_mines=NUM_MINES)
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=0.99, alpha=0.2)
    states = STATES_TO_EXPLAIN

    # Schritt 1: Q-Learning mit state-abhängigen gültigen Aktionen
    train(agent, env, TRAIN_STEPS)

    # Schritt 2: mögliche Minenlagen für die Zielzustände sammeln
    # (setzt env.instances, das Minesweeper.reset(state=...) zum gezielten
    #  Zurücksetzen auf einen Zielzustand braucht — siehe local_sverl Rollouts)
    find_states_minesweeper(agent, env, states, FIND_STATES_STEPS)

    # Negative Q-Werte (vom Treffen einer Mine) auf -1 kappen, damit die
    # Policy nicht versehentlich eine "am wenigsten schlechte Mine" bevorzugt —
    # gleiche Vorgehensweise wie im Haupttraining (minesweeper/run.py).
    true_q_table = copy.deepcopy(agent.Q_table)
    for state, values in agent.Q_table.items():
        agent.Q_table[state][values < 0] = -1

    # Schritt 3: Policy aus der bereinigten Q-Tabelle, dann Original wiederherstellen
    agent.get_policy(env.valid_dict)
    agent.Q_table = copy.deepcopy(true_q_table)
    agent.get_value_table(env.valid_dict)

    # Schritt 4: Besuchshäufigkeit je Brettzustand unter der gelernten Policy
    state_dist = get_state_dist(agent, env, STATE_SAMPLES)

    # Schritt 5: π_C — Policy, wenn nur Koalition C (bestimmte Felder) beobachtet wird
    F = np.arange(env.state_dim)
    ch = Characteristics(env, states)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states, env.valid_dict))
        for C in tqdm_label(F_not_i(F), "minesweeper pi_C")
    }
    # v_C — erwarteter Wert unter partieller Feature-Beobachtung
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), "minesweeper v_C")
    }

    # Schritt 6: Charakteristik v_C^π(s) — fast_local_sverl statt local_sverl
    # (Begründung siehe Modul-Docstring: 32.768 Koalitionen bei 16 Features
    #  sind mit episodenweisem Rollout pro Koalition nicht praktikabel)
    char_data = ch.fast_local_sverl_C_values(
        ch.pi_Cs, num_rolls=MS_NUM_ROLLS, valid_dict=env.valid_dict,
        multi_process=False, num_p=1,
    )

    # Schritt 7: kooperative Aufteilungsregeln → Beitrag pro Feld (Feature)
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
            results[m] = {MS_CHAR: calculators[m].run(char_data)}
        except Exception as exc:
            print(f"  WARN {m}: {exc}")

    for state in states:
        label = STATE_LABELS.get(tuple(state), "")
        print(f"  Zustand: {format_state(state)}  {label}")

    return {"results": results, "states": states}


def write_report(data: dict) -> None:
    """
    Schreibt ``AUSGABEWERTE.md``: Brettdarstellungen, Gesamt- und Detailtabelle.
    """
    results, states = data["results"], data["states"]

    def render_board(state):
        board = np.array(state, dtype=int).reshape(BOARD_HEIGHT, BOARD_LENGTH)
        return "\n".join(" ".join(f"{v:>2}" for v in row) for row in board)

    boards_md = []
    for i, state in enumerate(states):
        label = STATE_LABELS.get(tuple(state), "")
        boards_md.append(f"**Zustand {i + 1}** — {label}\n\n```\n{render_board(state)}\n```\n")

    body = [
        "# MINESWEEPER — Ausgabewert-Vergleich\n\n",
        f"4×4-Brett, {NUM_MINES} Minen. **16 Features** = Felder (0,0) … (3,3).\n\n",
        f"Charakteristik: **{MS_CHAR}** | Train: {TRAIN_STEPS:,} | Rolls/Zustand-Aktion: {MS_NUM_ROLLS:,}\n\n",
        "**Erklärte Zustände (Brett, -1=unaufgedeckt, 2=Mine aufgedeckt):**\n\n",
        "\n".join(boards_md),
        "\n## Gesamtvergleich (Mittel über beide Zustände)\n\n",
        mean_comparison_table(results, states, FEATURE_NAMES),
        "\n\n## Detailtabelle (pro Zustand)\n\n",
        comparison_table(results, states, FEATURE_NAMES),
        "\n",
    ]
    (OUT / "AUSGABEWERTE.md").write_text(
        "".join(body[:5]) + "\n\n![minesweeper](vergleich_gesamt.png)\n\n" + "".join(body[5:]),
        encoding="utf-8",
    )
    print(f"Bericht: {OUT / 'AUSGABEWERTE.md'}")


def main():
    """
    Ausführung: Ordner anlegen → Ausgabewerte berechnen → speichern und visualisieren.

    Erzeugt:
      - ``results.pkl``  — Rohdaten aller Methoden
      - ``vergleich_gesamt.png`` — Balken- und Heatmap-Vergleich der 16 Felder
      - ``AUSGABEWERTE.md`` — Textbericht mit Tabellen

    Hinweis: Wegen 16 Features (statt 2-9 bei den anderen Spielen) greift in
    ``save_comparison_plots`` automatisch der Heatmap-only-Zweig (>4 Features).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    data = run_minesweeper()

    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump(data["results"], f)

    save_comparison_plots(data["results"], data["states"], OUT, "minesweeper", FEATURE_NAMES)
    write_report(data)
    print(f"Fertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()