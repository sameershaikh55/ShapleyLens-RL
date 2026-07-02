import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleship.battleship_gym import BattleshipEnv
from q_agent_2 import Agent
from utils import find_states_battleship, train

WIDTH = 78

CHARACTERISTIC_MODE = "shapley_on_value"
METHOD_ORDER = ("shapley", "banzhaf", "nucleolus", "utopia-payoff", "gately", "tau")
METHOD_TITLES = {
    "shapley": "Shapley",
    "banzhaf": "Banzhaf (normiert, / 2^(n-1))",
    "nucleolus": "Nucleolus",
    "utopia-payoff": "Utopia-Payoff",
    "gately": "Gately",
    "tau": "Tau-Wert",
}


def _state_short(key, max_show=14):
    t = tuple(float(x) for x in key)
    if len(t) <= max_show:
        return "(" + ", ".join(f"{x:.0f}" for x in t) + ")"
    return f"({len(t)} Zellen, min={min(t):.0f}, max={max(t):.0f})"


def _print_state_block(state_key, per_feature_list, decimals=5):
    print(f"\n  Zustand  {_state_short(state_key)}")
    n_feat = len(per_feature_list)
    action_header = None
    rows_out = []
    for j in range(n_feat):
        v = per_feature_list[j]
        a = np.asarray(v, dtype=float).ravel()
        if a.size == 1:
            rows_out.append(f"    {j:3d}     {float(a[0]):.{decimals}f}")
        else:
            if action_header is None:
                action_header = "  ".join(f"a{k:>6}" for k in range(a.size))
            parts = "  ".join(f"{float(x):{decimals + 3}.{decimals}f}" for x in a)
            rows_out.append(f"    {j:3d}     {parts}")
    if action_header is not None:
        print(f"    {'feat':>4}  {action_header}")
        print("    " + "-" * (len(action_header) + 6))
    else:
        print(f"    {'feat':>4}  {'Wert':>{decimals + 3}}")
        print("    " + "-" * (decimals + 10))
    for line in rows_out:
        print(line)


def print_explanation_report(results, banzhaf_raw, char_modes, method_errors, grid_label="Battleship"):
    """Übersichtliche Terminal-Darstellung für alle Erklärwerte."""
    line = "=" * WIDTH
    print("\n" + line)
    print(f" {grid_label}: Shapley, Banzhaf, Nucleolus, Utopia, Gately, Tau")
    print(line)

    for char_name in char_modes:
        print("\n" + "-" * WIDTH)
        print(f" Charakteristik:  {char_name}")
        print("-" * WIDTH)

        for m in METHOD_ORDER:
            title = METHOD_TITLES[m]
            block = results.get(m) or {}
            vals = block.get(char_name) if isinstance(block, dict) else None
            if m in method_errors:
                print(f"\n  [{title}]  -  übersprungen: {method_errors[m]}")
                continue
            if not vals:
                print(f"\n  [{title}]  -  (keine Daten)")
                continue
            print(f"\n  > {title}")
            print("  " + "-" * (WIDTH - 4))
            for sk in sorted(vals.keys(), key=lambda k: (len(k), str(k))):
                _print_state_block(sk, vals[sk])

        br = (banzhaf_raw or {}).get(char_name)
        if br and "banzhaf_raw" not in method_errors:
            print(f"\n  > Banzhaf (roh, ohne / 2^(n-1))")
            print("  " + "-" * (WIDTH - 4))
            for sk in sorted(br.keys(), key=lambda k: (len(k), str(k))):
                _print_state_block(sk, br[sk])
        elif "banzhaf_raw" in method_errors:
            print(f"\n  [Banzhaf roh]  -  übersprungen: {method_errors['banzhaf_raw']}")

    print("\n" + line + "\n")


def render_pretty(state, rows, cols):
    """Darstellung: . unbekannt, X Treffer, o Fehlschuss."""
    grid = np.array(state).reshape(rows, cols)
    print("\nSpielfeld:")
    for row in grid:
        print(
            " ".join(
                [
                    "." if x == 0 else "X" if x == 1 else "o"
                    for x in row
                ]
            )
        )
    print()


def collect_sample_states(env, agent, n_episodes=200, max_states=4):
    """Ein paar Beobachtungs-Zustände aus Rollouts sammeln (für Erklärung)."""
    seen = []
    for _ in range(n_episodes):
        state, info = env.reset()
        for _ in range(env.state_dim + 5):
            if len(seen) >= max_states:
                return np.array(seen[:max_states], dtype=np.float64)
            row = np.asarray(state, dtype=np.float64).flatten()
            if not any((row == s).all() for s in seen):
                seen.append(row.copy())
            action = agent.choose_action(state, info)
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    return np.array(seen[:max_states], dtype=np.float64) if seen else np.zeros((1, env.state_dim))


def battleship_init():
    env = BattleshipEnv(seed=0, render_mode=None)
    agent = Agent(env.state_dim, env.num_actions, epsilon=1.0, gamma=0.95, alpha=0.1)
    states_to_explain = np.zeros((1, env.state_dim), dtype=np.float64)
    return env, agent, states_to_explain


def battleship_get_characteristic_modes():
    return [CHARACTERISTIC_MODE]


def battleship_run(
    env=None,
    agent=None,
    states_to_explain=None,
    methods=None,
    normalized=True,
    scale_factor=1.0,
):
    """Train Battleship and compute explainer values via the adaptive pipeline."""
    from explainer import Explainer

    print(f"Training ({env.rows}x{env.cols} = {env.state_dim} Merkmale)...\n")
    train(agent, env, int(5e4))

    agent.epsilon = 0.0
    states_to_explain = collect_sample_states(env, agent, n_episodes=400, max_states=2)
    instances = find_states_battleship(agent, env, states_to_explain, max_steps=500_000)
    if not instances:
        raise RuntimeError("Keine Battleship-Instanzen gefunden")

    if len(instances) < len(states_to_explain):
        states_to_explain = np.stack(
            [np.array(k, dtype=np.float64) for k in instances.keys()], axis=0
        )

    agent.get_policy()
    agent.get_value_table()

    if methods is None:
        methods = list(METHOD_ORDER)

    explainer = Explainer(
        env, agent, states_to_explain, instances=instances,
        normalize=normalized, scale_factor=scale_factor,
    )
    explainer.compute_state_dist(sample_size=5e4)

    adaptive_results = explainer.run_values(
        methods=methods,
        normalized=normalized,
        characteristic_mode=CHARACTERISTIC_MODE,
    )
    meta = adaptive_results.pop("_meta", {})

    return {
        "states_to_explain": states_to_explain,
        "results": adaptive_results,
        "method_errors": meta.get("method_errors", {}),
    }


if __name__ == "__main__":
    assert BattleshipEnv.NUM_FEATURES == 40
    assert BattleshipEnv.NUM_ROWS == 5
    assert BattleshipEnv.NUM_COLS == 8

    env, agent, _ = battleship_init()
    assert env.state_dim == BattleshipEnv.NUM_FEATURES
    grid_label = f"Battleship ({env.rows}x{env.cols})"

    payload = battleship_run(
        env=env,
        agent=agent,
        methods=list(METHOD_ORDER),
        normalized=True,
    )

    print_explanation_report(
        payload["results"],
        {},
        [CHARACTERISTIC_MODE],
        payload["method_errors"],
        grid_label=grid_label,
    )

    results = payload["results"]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for method, method_results in results.items():
        for name, values in method_results.items():
            path = os.path.join(out_dir, f"{method}_{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(values, f)

"""    for name, values in banzhaf_raw.items():
        path = os.path.join(out_dir, f"battleship_demo_banzhaf_unnormalized_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(values, f)"""
