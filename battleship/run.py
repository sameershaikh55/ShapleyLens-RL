import os
import sys
import pickle

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleship_gym import BattleshipEnv
from explainer import Explainer
from q_agent_1 import Agent as ShapleyAgent
from q_agent_2 import Agent as TrainAgent
from utils import find_states_battleship, train

WIDTH = 78

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


def print_explanation_report(results, banzhaf_raw, char_modes, method_errors):
    """Übersichtliche Terminal-Darstellung für alle Erklärwerte."""
    line = "=" * WIDTH
    print("\n" + line)
    print(" Explainer-Demo (3x3): Shapley, Banzhaf, Nucleolus, Utopia, Gately, Tau")
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


def run_explainer_demo():
    """
    Kleines Raster (3x3): Shapley, Banzhaf, Nucleolus, Utopia, Gately, Tau - tabellarisch + *.pkl.
    """
    demo_env = BattleshipEnv(rows=3, cols=3, ship_sizes=[2, 1, 1], seed=0, render_mode=None)
    demo_agent = TrainAgent(demo_env.state_dim, demo_env.num_actions)
    train(demo_agent, demo_env, int(5e4))

    sv = ShapleyAgent(demo_env.state_dim, demo_env.num_actions, epsilon=0.0, gamma=0.95, alpha=0.2)
    sv.Q_table = demo_agent.Q_table
    sv.get_policy()
    sv.get_value_table()

    states_to_explain = collect_sample_states(demo_env, demo_agent, n_episodes=400, max_states=2)
    instances = find_states_battleship(demo_agent, demo_env, states_to_explain, max_steps=500_000)
    if not instances:
        return
    if len(instances) < len(states_to_explain):
        states_to_explain = np.stack([np.array(k, dtype=np.float64) for k in instances.keys()], axis=0)

    explainer = Explainer(demo_env, sv, states_to_explain, instances=instances)
    explainer.compute_state_dist(sample_size=50_000)
    explainer.compute_pi_Cs()
    explainer.compute_v_Cs()

    modes = ["shapley_on_policy", "shapley_on_value"]
    characteristics = explainer.compute_characteristics(
        modes, num_rolls=1, multi_process=False, num_p=1
    )
    method_errors = {}
    results = {}
    for m in METHOD_ORDER:
        try:
            results[m] = explainer.run_values(
                characteristics, methods=(m,), normalized=True
            ).get(m, {})
        except Exception as exc:
            method_errors[m] = str(exc)
            results[m] = {}

    banzhaf_raw = {}
    try:
        banzhaf_raw = explainer.run_values(
            characteristics, methods=("banzhaf",), normalized=False
        ).get("banzhaf", {})
    except Exception as exc:
        method_errors["banzhaf_raw"] = str(exc)

    print_explanation_report(results, banzhaf_raw, modes, method_errors)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for method, method_results in results.items():
        for name, values in method_results.items():
            path = os.path.join(out_dir, f"battleship_demo_{method}_{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(values, f)
    for name, values in banzhaf_raw.items():
        path = os.path.join(out_dir, f"battleship_demo_banzhaf_unnormalized_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(values, f)


if __name__ == "__main__":
    assert BattleshipEnv.NUM_FEATURES == 40

    env = BattleshipEnv(render_mode="human")
    assert env.state_dim == BattleshipEnv.NUM_FEATURES

    agent = TrainAgent(env.state_dim, env.num_actions)

    episodes = 5000
    print("Training (5x8 = 40 Merkmale)...\n")

    rewards_history = []
    for ep in tqdm(range(episodes), desc="Training", ncols=100):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.choose_action(state, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done, info)
            state = next_state
            total_reward += reward
        rewards_history.append(total_reward)
        if ep % 500 == 0:
            tqdm.write(
                f"Episode {ep} | Avg Reward (letzte 100): {np.mean(rewards_history[-100:]):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\nTestspiel (epsilon=0)\n")
    agent.epsilon = 0.0
    state, info = env.reset()
    done = False
    render_pretty(state, env.rows, env.cols)
    step_counter = 0
    while not done:
        action = agent.choose_action(state, info)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        render_pretty(state, env.rows, env.cols)
        print(f"Schritt: {step_counter} | Aktion: {action} | Reward: {reward}")
        step_counter += 1

    print("\nSpiel beendet.")
    if info.get("result") == "win":
        print("Gewonnen.")
    else:
        print("Verloren oder abgebrochen.")

    run_explainer_demo()
