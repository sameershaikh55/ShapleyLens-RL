"""
Ausgabewert-Auswertung: Taxi (Gymnasium Taxi-v3).

Berechnet Feature-Beiträge (Shapley, Banzhaf, …) für 4 ausgewählte Zustände.
4 Features = [taxi_row, taxi_col, passenger_loc, destination]. Optimale Policy
per Value Iteration (kein Q-Learning-Training, exakte Lösung des Tabellen-MDP).

Ergebnisse → ``Ausgabewert/taxi/``

    python ausgabewert_taxi.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    import gym  # type: ignore[import]
except ImportError:
    import gymnasium as gym

from characteristics import Characteristics
from q_agent_1 import Agent
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
from taxi.taxi_wrap import FactoredState
from utils import F_not_i, get_state_dist, find_states_taxi, tqdm_label, value_iteration

OUT = ROOT / "Ausgabewert" / "taxi"
FEATURE_NAMES = ["taxi_row", "taxi_col", "passenger_loc", "destination"]
GAMMA = 0.99            # Diskontierungsfaktor der Value Iteration
STATE_SAMPLES = 50_000  # Stichproben für die Zustandsverteilung p^π(s)

# Taxi-Episoden sind viel länger als in den Grid Worlds (mehr Schritte bis
# Pickup+Drop-Off), daher wird die geteilte NUM_ROLLS-Konstante aus
# ausgabewert_grid_worlds hier lokal überschrieben, um die Laufzeit der
# Monte-Carlo-Rollouts (16 Koalitionen × 4 Zustände) im Rahmen zu halten.
TAXI_NUM_ROLLS = 300

# Vier handverlesene Zustände, die unterschiedliche Phasen der Taxi-Aufgabe
# abdecken (analog zu den fest gewählten Zuständen in gwa/gwb/gwc).
#   passenger_loc: 0=R, 1=G, 2=Y, 3=B, 4=im Taxi
#   destination:   0=R, 1=G, 2=Y, 3=B
STATES_TO_EXPLAIN = np.array([
    [0, 0, 0, 1],  # Taxi an Pickup R, Passagier wartet an R, Ziel G — vor Abholung
    [0, 0, 4, 1],  # Taxi an R, Passagier bereits im Taxi, Ziel G — direkt nach Abholung
    [0, 4, 4, 1],  # Taxi an Ziel G, Passagier im Taxi, Ziel G — kurz vor Absetzen
    [2, 2, 2, 3],  # Taxi mittig im Gitter, Passagier an Y, Ziel B — generische Navigation
], dtype=float)

STATE_LABELS = {
    (0.0, 0.0, 0.0, 1.0): "Vor Abholung (an R)",
    (0.0, 0.0, 4.0, 1.0): "Nach Abholung (an R, Passagier im Taxi)",
    (0.0, 4.0, 4.0, 1.0): "Vor Absetzen (an G, Passagier im Taxi)",
    (2.0, 2.0, 2.0, 3.0): "Generische Navigation (Mitte des Gitters)",
}


def run_taxi() -> dict:
    """
    Hauptpipeline: optimale Policy → Charakteristikwerte → Ausgabewerte.

    Schritte:
      1. Value Iteration auf dem Taxi-MDP (exakt, kein Sampling-Training)
      2. Wertfunktion V(s) aus Q-Tabelle ableiten
      3. Vier feste Zustände als Erklärziele (Pickup/Drop-Off-Phasen)
      4. Stationäre Zustandsverteilung p^π(s) unter π* schätzen
      5. Für jede Feature-Koalition C: partielle Policies π_C und Werte v_C
      6. Local SVERL (Monte-Carlo): erwarteter Return pro Koalition und Zustand
      7. Shapley, Banzhaf, Nucleolus, Tau, Utopia, Gately anwenden

    Returns:
        dict mit ``results`` (Ausgabewerte pro Methode) und ``states`` (erklärte Zustände)
    """
    print(f"\n{'='*60}\n  TAXI\n{'='*60}")

    try:
        env = FactoredState(gym.make("Taxi-v3"))
    except Exception:
        # Neuere Gymnasium-Versionen haben Taxi-v3 zugunsten von Taxi-v4
        # entfernt; die zugrunde liegende MDP- und Beobachtungs-API ist
        # identisch (Discrete(500) Zustände, Discrete(6) Aktionen, gleiches
        # decode()-Schema), daher genügt ein einfacher Fallback.
        env = FactoredState(gym.make("Taxi-v4"))
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.0, gamma=GAMMA, alpha=0.2)
    states = STATES_TO_EXPLAIN

    # Schritt 1–2: exakte optimale Policy (Tabellen-MDP), kein Sampling-Training
    agent.Q_table, agent.policy = value_iteration(env, gamma=GAMMA)
    agent.get_value_table()

    # Schritt 3: env-Instanzen für die vier festen Zustände einsammeln
    # (lokales SVERL braucht eine konkrete env, um Episoden ab diesem Zustand
    #  auszurollen — Taxi kann nicht direkt in einen beliebigen Zustand versetzt werden)
    instances = find_states_taxi(agent, env, states)

    # Schritt 4: Besuchshäufigkeit je Zustand unter der optimalen Policy
    state_dist = get_state_dist(agent, env, STATE_SAMPLES)

    # Schritt 5: π_C — Policy, wenn nur Koalition C der Features beobachtet wird
    F = np.arange(env.state_dim)
    ch = Characteristics(env, states, instances=instances)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states))
        for C in tqdm_label(F_not_i(F), "taxi pi_C")
    }
    # v_C — erwarteter Wert unter partieller Feature-Beobachtung
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), "taxi v_C")
    }

    # Schritt 6: Charakteristik v_C^π(s) — Rollouts mit TAXI_NUM_ROLLS pro Koalition
    char_data = ch.local_sverl_C_values(TAXI_NUM_ROLLS, ch.pi_Cs, multi_process=False, num_p=1)

    # Schritt 7: kooperative Aufteilungsregeln → Beitrag pro Feature
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

    for state in states:
        label = STATE_LABELS.get(tuple(state), "")
        print(f"  Zustand: {format_state(state)}  {label}")

    return {"results": results, "states": states}


def write_report(data: dict) -> None:
    """
    Schreibt ``AUSGABEWERTE.md``: Zustandsbeschreibungen, Gesamt- und Detailtabelle.
    """
    results, states = data["results"], data["states"]
    state_lines = "\n".join(
        f"- `{format_state(s)}` — {STATE_LABELS.get(tuple(s), '')}" for s in states
    )
    body = [
        "# TAXI — Ausgabewert-Vergleich\n\n",
        "Gymnasium **Taxi-v3** (faktorisierter Zustand). "
        "**4 Features**: taxi_row, taxi_col, passenger_loc, destination.\n\n",
        f"Value Iteration (γ={GAMMA}) | Charakteristik: **{CHAR}** | Rolls: {TAXI_NUM_ROLLS:,}\n\n",
        "**Erklärte Zustände:**\n\n",
        state_lines,
        "\n\n## Gesamtvergleich (Mittel über alle vier Zustände)\n\n",
        mean_comparison_table(results, states, FEATURE_NAMES),
        "\n\n## Detailtabelle (pro Zustand)\n\n",
        comparison_table(results, states, FEATURE_NAMES),
        "\n",
    ]
    (OUT / "AUSGABEWERTE.md").write_text(
        "".join(body[:6]) + "\n\n![taxi](vergleich_gesamt.png)\n\n" + "".join(body[6:]),
        encoding="utf-8",
    )
    print(f"Bericht: {OUT / 'AUSGABEWERTE.md'}")


def main():
    """
    Ausführung: Ordner anlegen → Ausgabewerte berechnen → speichern und visualisieren.

    Erzeugt:
      - ``results.pkl``  — Rohdaten aller Methoden
      - ``vergleich_gesamt.png`` — Balken- und Heatmap-Vergleich der 4 Features
      - ``AUSGABEWERTE.md`` — Textbericht mit Tabellen
    """
    OUT.mkdir(parents=True, exist_ok=True)
    data = run_taxi()

    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump(data["results"], f)

    save_comparison_plots(data["results"], data["states"], OUT, "taxi", FEATURE_NAMES)
    write_report(data)
    print(f"Fertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()