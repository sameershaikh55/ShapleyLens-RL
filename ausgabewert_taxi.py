"""
Ausgabewert-Auswertung: Taxi-v3.

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
from utils import F_not_i, find_states_taxi, get_state_dist, tqdm_label, value_iteration

OUT = ROOT / "Ausgabewert" / "taxi"
FEATURE_NAMES = ["taxi_r", "taxi_c", "pass", "dest"]
GAMMA = 0.99
NUM_ROLLS = 100
STATE_SAMPLES = 50_000

# Standard-Zustände aus taxi/run.py
DEFAULT_STATES = np.array([[0, 3, 3, 1], [0, 3, 4, 3]], dtype=float)


def run_taxi() -> dict:
    print(f"\n{'='*60}\n  TAXI\n{'='*60}")

    env = FactoredState(gym.make("Taxi-v3"))
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.1, gamma=GAMMA, alpha=0.2)
    agent.Q_table, agent.policy = value_iteration(env, gamma=GAMMA)
    agent.get_value_table()

    states = DEFAULT_STATES
    print(f"  Erklärte Zustände: {len(states)}")

    instances = find_states_taxi(agent, env, states)
    state_dist = get_state_dist(agent, env, STATE_SAMPLES)

    F = np.arange(env.state_dim)
    ch = Characteristics(env, states, instances=instances)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states))
        for C in tqdm_label(F_not_i(F), "taxi pi_C")
    }
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), "taxi v_C")
    }
    char_data = ch.local_sverl_C_values(NUM_ROLLS, ch.pi_Cs, multi_process=False, num_p=1)

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

    for s in states:
        print(f"  Zustand: {format_state(s)}")
    return {"results": results, "states": states}


def write_report(data: dict) -> None:
    results, states = data["results"], data["states"]
    body = [
        "# TAXI — Ausgabewert-Vergleich\n\n",
        "Gym **Taxi-v3**, faktorisierter Zustand (4 Features).\n\n",
        f"Value Iteration (γ={GAMMA}) | Charakteristik: **{CHAR}** | Rolls: {NUM_ROLLS}\n\n",
        "**Features:** taxi_r, taxi_c (Taxi-Position), pass (Passagier), dest (Ziel)\n\n",
        f"**Erklärte Zustände:** {len(states)}\n\n",
        "## Gesamtvergleich (Mittel über alle Zustände)\n\n",
        mean_comparison_table(results, states, FEATURE_NAMES),
        "\n\n## Detailtabelle (pro Zustand)\n\n",
        comparison_table(results, states, FEATURE_NAMES),
        "\n",
    ]
    (OUT / "AUSGABEWERTE.md").write_text(
        "".join(body[:7]) + "\n\n![taxi](vergleich_gesamt.png)\n\n" + "".join(body[7:]),
        encoding="utf-8",
    )
    print(f"Bericht: {OUT / 'AUSGABEWERTE.md'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = run_taxi()

    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump(data["results"], f)

    save_comparison_plots(data["results"], data["states"], OUT, "taxi", FEATURE_NAMES)
    write_report(data)
    print(f"Fertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()
