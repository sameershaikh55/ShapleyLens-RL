"""
Ausgabewert-Auswertung: FrozenLake.

Berechnet Feature-Beiträge (Shapley, Banzhaf, …) für Gym FrozenLake-v1.
2 Features: Zeile (row) und Spalte (col). Alle nicht-terminalen Gitterfelder
werden erklärt. Optimale Policy per Value Iteration (kein Q-Learning-Training).

Ergebnisse → ``Ausgabewert/frozen_lake/``

    python ausgabewert_frozen_lake.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from characteristics import Characteristics
from frozen_lake.frozen_lake_wrap import FrozenLakeWrap
from q_agent_1 import Agent
from ausgabewert_grid_worlds import (
    CHAR,
    METHODS,
    NUM_ROLLS,
    comparison_table,
    mean_comparison_table,
    save_comparison_plots,
)
from shapley import Shapley
from banzhaf import Banzhaf
from nucleolus import Nucleolus
from tau import TauValue
from utopia_payoff import UtopiaPayoff
from gately import Gately
from utils import F_not_i, get_state_dist, tqdm_label, value_iteration

OUT = ROOT / "Ausgabewert" / "frozen_lake"
FEATURE_NAMES = ["row", "col"]  # faktorisierter Zustand: (Zeile, Spalte)
GAMMA = 0.99                    # Diskontierungsfaktor der Value Iteration
STATE_SAMPLES = 10_000          # Stichproben für die Zustandsverteilung p^π(s)


def run_frozen_lake() -> dict:
    """
    Hauptpipeline: optimale Policy → Charakteristikwerte → Ausgabewerte.

    Schritte:
      1. Value Iteration auf dem 4×4-Gitter (slippery) → Q* und π*
      2. Wertfunktion V(s) aus Q-Tabelle ableiten
      3. Alle nicht-terminalen Zustände als Erklärziele wählen
      4. Stationäre Zustandsverteilung p^π(s) unter π* schätzen
      5. Für jede Feature-Koalition C: partielle Policies π_C und Werte v_C
      6. Local SVERL (Monte-Carlo): erwarteter Return pro Koalition und Zustand
      7. Shapley, Banzhaf, Nucleolus, Tau, Utopia, Gately anwenden

    Returns:
        dict mit ``results`` (Ausgabewerte pro Methode) und ``states`` (alle erklärten Zustände)
    """
    print(f"\n{'='*60}\n  FROZEN LAKE\n{'='*60}")

    env = FrozenLakeWrap(is_slippery=True)
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.0, gamma=GAMMA, alpha=0.2)

    # Schritt 1–2: exakte optimale Policy (Tabellen-MDP), kein Sampling-Training
    agent.Q_table, agent.policy = value_iteration(env, gamma=GAMMA)
    agent.get_value_table()

    # Schritt 3: Loch- und Zielfelder sind terminal → nur sichere Startfelder
    states = np.array(list(agent.policy.keys()), dtype=float)
    print(f"  Nicht-terminale Zustände: {len(states)}")

    # Schritt 4: Besuchshäufigkeit je (row, col) unter der optimalen Policy
    state_dist = get_state_dist(agent, env, STATE_SAMPLES)

    # Schritt 5: π_C — Policy, wenn nur Koalition C (row/col) beobachtet wird
    F = np.arange(env.state_dim)
    ch = Characteristics(env, states)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states))
        for C in tqdm_label(F_not_i(F), "frozen_lake pi_C")
    }
    # v_C — erwarteter Wert unter partieller Feature-Beobachtung
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), "frozen_lake v_C")
    }

    # Schritt 6: Charakteristik v_C^π(s) — Rollouts mit NUM_ROLLS pro Koalition
    char_data = ch.local_sverl_C_values(NUM_ROLLS, ch.pi_Cs, multi_process=False, num_p=1)

    # Schritt 7: kooperative Aufteilungsregeln → Beitrag von row und col
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

    return {"results": results, "states": states}


def write_report(data: dict) -> None:
    """
    Schreibt ``AUSGABEWERTE.md``: Gesamt- und Detailtabelle.

    ``mean_comparison_table`` — Mittelwert von row/col über alle 11 Zustände.
    ``comparison_table`` — Ausgabewerte pro Zustand und Feature.
    """
    results, states = data["results"], data["states"]
    body = [
        "# FROZEN LAKE — Ausgabewert-Vergleich\n\n",
        "Gym **FrozenLake-v1** (4×4, slippery). Features: **row**, **col**.\n\n",
        f"Value Iteration (γ={GAMMA}) | Charakteristik: **{CHAR}** | Rolls: {NUM_ROLLS:,}\n\n",
        f"**Erklärte Zustände:** {len(states)} (nicht-terminal)\n\n",
        "## Gesamtvergleich (Mittel über alle Zustände)\n\n",
        mean_comparison_table(results, states, FEATURE_NAMES),
        "\n\n## Detailtabelle (pro Zustand)\n\n",
        comparison_table(results, states, FEATURE_NAMES),
        "\n",
    ]
    (OUT / "AUSGABEWERTE.md").write_text(
        "".join(body[:8]) + "\n\n![frozen_lake](vergleich_gesamt.png)\n\n" + "".join(body[8:]),
        encoding="utf-8",
    )
    print(f"Bericht: {OUT / 'AUSGABEWERTE.md'}")


def main():
    """
    Ausführung: Ordner anlegen → Ausgabewerte berechnen → speichern und visualisieren.

    Erzeugt:
      - ``results.pkl``  — Rohdaten aller Methoden
      - ``vergleich_gesamt.png`` — Balken- und Heatmap-Vergleich row vs. col
      - ``AUSGABEWERTE.md`` — Textbericht mit Tabellen
    """
    OUT.mkdir(parents=True, exist_ok=True)
    data = run_frozen_lake()

    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump(data["results"], f)

    save_comparison_plots(
        data["results"], data["states"], OUT, "frozen_lake", FEATURE_NAMES,
    )
    write_report(data)
    print(f"Fertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()
