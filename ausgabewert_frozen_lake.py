"""
Ausgabewert-Auswertung: FrozenLake.

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
FEATURE_NAMES = ["row", "col"]
GAMMA = 0.99
STATE_SAMPLES = 10_000


def run_frozen_lake() -> dict:
    print(f"\n{'='*60}\n  FROZEN LAKE\n{'='*60}")

    env = FrozenLakeWrap(is_slippery=True)
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.0, gamma=GAMMA, alpha=0.2)
    agent.Q_table, agent.policy = value_iteration(env, gamma=GAMMA)
    agent.get_value_table()

    states = np.array(list(agent.policy.keys()), dtype=float)
    print(f"  Nicht-terminale Zustände: {len(states)}")

    state_dist = get_state_dist(agent, env, STATE_SAMPLES)
    F = np.arange(env.state_dim)
    ch = Characteristics(env, states)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states))
        for C in tqdm_label(F_not_i(F), "frozen_lake pi_C")
    }
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), "frozen_lake v_C")
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

    return {"results": results, "states": states}


def write_report(data: dict) -> None:
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
