import os
import sys
import pickle

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleship import Battleship
from explainer import Explainer
from q_agent_1 import Agent as ShapleyAgent
from q_agent_2 import Agent as TrainAgent
from utils import find_states_battleship, train


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
    Kleines Raster (3x3): exakte Shapley/Banzhaf über alle Koalitionen sind machbar.
    Das Standardspiel 5x8 hat 40 Merkmale -> 2^40 Koalitionen (nur Training/Test oben).
    """
    demo_env = Battleship(rows=3, cols=3, ship_sizes=[2, 1, 1], seed=0)
    demo_agent = TrainAgent(demo_env.state_dim, demo_env.num_actions)
    train(demo_agent, demo_env, int(5e4))

    sv = ShapleyAgent(demo_env.state_dim, demo_env.num_actions, epsilon=0.0, gamma=0.95, alpha=0.2)
    sv.Q_table = demo_agent.Q_table
    sv.get_policy()
    sv.get_value_table()

    states_to_explain = collect_sample_states(demo_env, demo_agent, n_episodes=400, max_states=2)
    instances = find_states_battleship(demo_agent, demo_env, states_to_explain, max_steps=500_000)
    if not instances:
        print("Explainer-Demo: keine Environment-Instanzen gefunden (übersprungen).")
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
    results = explainer.run_values(characteristics, methods=("shapley", "banzhaf"), normalized=True)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for method, method_results in results.items():
        for name, values in method_results.items():
            print(f"\n=== {method.capitalize()} ({name}) ===")
            print(values)
            path = os.path.join(out_dir, f"battleship_demo_{method}_{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(values, f)


if __name__ == "__main__":
    assert Battleship.NUM_FEATURES == 40

    env = Battleship()
    assert env.state_dim == Battleship.NUM_FEATURES

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
    print(f"Treffer: {env.hits} | Schritte: {env.steps}")

    print(
        "\nHinweis: Exakte Koalitionsauswertung für alle 40 Zellen erfordert 2^40 Charakteristiken."
        "\nExplainer-Demo mit 3x3-Raster:"
    )
    run_explainer_demo()
