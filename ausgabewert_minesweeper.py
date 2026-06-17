"""
Ausgabewert-Auswertung: Minesweeper (4×4, 2 Minen).

Ergebnisse → ``Ausgabewert/minesweeper/``

    python ausgabewert_minesweeper.py
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from characteristics import Characteristics
from q_agent_2 import Agent, PolicyDict
from ausgabewert_grid_worlds import (
    CHAR,
    METHODS,
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
from minesweeper.minesweeper import Minesweeper
from utils import F_not_i, find_states_minesweeper, get_state_dist, train, tqdm_label

OUT = ROOT / "Ausgabewert" / "minesweeper"
WORKER_CTX = OUT / "worker_ctx.pkl"
BOARD_SIZE = 4
FEATURE_NAMES = [f"({r},{c})" for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
TRAIN_STEPS = 500_000
FIND_STEPS = 300_000
STATE_SAMPLES = 50_000
NUM_ROLLS = 50
NUM_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))

# Standard-Bretter aus minesweeper/run.py (-1 = verdeckt)
DEFAULT_STATES = np.array(
    [
        [
            0, 0, 1, -1,
            0, 1, 2, -1,
            0, 1, -1, -1,
            0, 1, 1, 1,
        ],
        [
            0, 0, 1, -1,
            0, 1, 2, -1,
            0, 1, -1, 2,
            0, 1, 1, 1,
        ],
    ],
    dtype=int,
)

_WORKER: dict = {}


def _save_worker_ctx(
    path: Path,
    agent: Agent,
    state_dist: dict,
    states: np.ndarray,
    env: Minesweeper,
) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "state_dim": agent.state_dim,
                "num_actions": agent.num_actions,
                "epsilon": agent.epsilon,
                "gamma": agent.gamma,
                "alpha": agent.alpha,
                "Q_table": {k: v.copy() for k, v in agent.Q_table.items()},
                "policy_items": {k: v.copy() for k, v in agent.policy.items()},
                "value_table": dict(agent.value_table),
                "state_dist": dict(state_dist),
                "states": states,
                "board_size": env.length,
                "num_mines": env.num_mines,
            },
            f,
        )


def _init_worker(ctx_path: str) -> None:
    global _WORKER
    with open(ctx_path, "rb") as f:
        data = pickle.load(f)
    env = Minesweeper(
        length=data["board_size"],
        height=data["board_size"],
        num_mines=data["num_mines"],
    )
    agent = Agent(
        data["state_dim"],
        data["num_actions"],
        epsilon=data["epsilon"],
        gamma=data["gamma"],
        alpha=data["alpha"],
    )
    for state, values in data["Q_table"].items():
        agent.Q_table[state] = values
    agent.policy = PolicyDict()
    agent.policy.valid_dict = env.valid_dict
    agent.policy.num_actions = agent.num_actions
    for state, probs in data["policy_items"].items():
        agent.policy[state] = probs
    agent.value_table = data["value_table"]
    _WORKER = {
        "agent": agent,
        "state_dist": data["state_dist"],
        "states": data["states"],
        "valid_dict": env.valid_dict,
    }


def _compute_pi_C(C: list) -> tuple:
    agent = _WORKER["agent"]
    pi = agent.get_pi_C(C, _WORKER["state_dist"], _WORKER["states"], _WORKER["valid_dict"])
    return tuple(C), dict(pi)


def _compute_v_C(C: list) -> tuple:
    agent = _WORKER["agent"]
    v = agent.get_v_C(C, _WORKER["state_dist"], _WORKER["states"])
    return tuple(C), v


def format_board(state) -> str:
    board = np.asarray(state, dtype=int).reshape(BOARD_SIZE, BOARD_SIZE)
    lines = []
    for row in board:
        cells = []
        for v in row:
            cells.append("B" if v == -1 else str(v))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def prepare_policy(agent: Agent, env: Minesweeper) -> None:
    """Q-Werte < 0 für Policy maskieren (wie minesweeper/run.py)."""
    true_q = copy.deepcopy(agent.Q_table)
    for state, values in agent.Q_table.items():
        agent.Q_table[state][values < 0] = -1
    agent.get_policy(env.valid_dict)
    agent.Q_table = true_q
    agent.get_value_table(env.valid_dict)


def _parallel_coalitions(
    label: str,
    coalitions: list,
    fn_name: str,
    ctx_path: Path,
) -> dict:
    results = {}
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(str(ctx_path),),
    ) as pool:
        compute = _compute_pi_C if fn_name == "pi_C" else _compute_v_C
        futures = {pool.submit(compute, C): C for C in coalitions}
        for fut in tqdm_label(as_completed(futures), label, total=len(futures)):
            key, value = fut.result()
            results[key] = value
    return results


def run_minesweeper() -> dict:
    print(f"\n{'='*60}\n  MINESWEEPER\n{'='*60}")
    print(f"  Worker: {NUM_WORKERS}")

    env = Minesweeper(length=BOARD_SIZE, height=BOARD_SIZE, num_mines=2)
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=0.99, alpha=0.2)
    states = DEFAULT_STATES.copy()

    train(agent, env, TRAIN_STEPS)
    find_states_minesweeper(agent, env, states, FIND_STEPS)

    missing = [s for s in states if tuple(s) not in getattr(env, "instances", {})]
    if missing:
        raise RuntimeError(
            f"Keine Minen-Instanzen für {len(missing)} Zustand(e) gefunden. "
            f"FIND_STEPS erhöhen (aktuell {FIND_STEPS:,})."
        )

    prepare_policy(agent, env)

    print(f"  Erklärte Zustände: {len(states)}")
    for i, s in enumerate(states):
        n_inst = env.instances[tuple(s)][2]
        print(f"  Brett {i + 1}: {n_inst} mögliche Minen-Konfiguration(en)")

    state_dist = get_state_dist(agent, env, STATE_SAMPLES)
    coalitions = F_not_i(np.arange(env.state_dim))
    print(f"  Koalitionen: {len(coalitions):,}")

    _save_worker_ctx(WORKER_CTX, agent, state_dist, states, env)

    ch = Characteristics(env, states)
    ch.pi_Cs = _parallel_coalitions("minesweeper pi_C", coalitions, "pi_C", WORKER_CTX)
    ch.v_Cs = _parallel_coalitions("minesweeper v_C", coalitions, "v_C", WORKER_CTX)

    char_data = ch.fast_local_sverl_C_values(
        ch.pi_Cs,
        num_rolls=NUM_ROLLS,
        valid_dict=env.valid_dict,
        multi_process=True,
        num_p=NUM_WORKERS,
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

    return {"results": results, "states": states}


def write_report(data: dict) -> None:
    results, states = data["results"], data["states"]
    board_blocks = []
    for i, s in enumerate(states):
        board_blocks.append(f"**Brett {i + 1}** (`B` = verdeckt):\n\n```\n{format_board(s)}\n```\n")

    before_img = [
        "# MINESWEEPER — Ausgabewert-Vergleich\n\n",
        f"Minesweeper **{BOARD_SIZE}×{BOARD_SIZE}**, 2 Minen. **16 Features** = Felder (0,0) … (3,3).\n\n",
        f"Q-Learning (Train: {TRAIN_STEPS:,}, γ=0.99) | Charakteristik: **{CHAR}** "
        f"(fast local SVERL) | Rolls: {NUM_ROLLS}\n\n",
        f"**Erklärte Zustände:** {len(states)}\n\n",
        "".join(board_blocks),
        "\n",
        "## Gesamtvergleich (Mittel über alle Zustände)\n\n",
        mean_comparison_table(results, states, FEATURE_NAMES),
    ]
    after_img = [
        "\n\n## Detailtabelle (pro Zustand)\n\n",
        comparison_table(results, states, FEATURE_NAMES),
        "\n",
    ]
    (OUT / "AUSGABEWERTE.md").write_text(
        "".join(before_img) + "\n\n![minesweeper](vergleich_gesamt.png)\n\n" + "".join(after_img),
        encoding="utf-8",
    )
    print(f"Bericht: {OUT / 'AUSGABEWERTE.md'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = run_minesweeper()

    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump(data["results"], f)

    save_comparison_plots(
        data["results"], data["states"], OUT, "minesweeper", FEATURE_NAMES,
    )
    write_report(data)
    print(f"Fertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()
