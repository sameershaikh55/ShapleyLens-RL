import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from q_agent_1 import Agent
from gwd import Grid
from utils import train, get_state_dist
from explainer import Explainer

if __name__ == '__main__':
    env = Grid(10, 10, 20)
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.1, gamma=1, alpha=0.2)

    # ------------------------------------------------- TRAIN
    train(agent, env, 1e7)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy()

    # ------------------------------------------------- EXPLAINER
    state_dist = get_state_dist(agent, env, 1e7)
    states_to_explain = np.unique(np.array(list(state_dist)), axis=0)
    explainer = Explainer(env, agent, states_to_explain)
    explainer.state_dist = state_dist
    explainer.compute_pi_Cs()
    explainer.compute_v_Cs()
    characteristic_modes = ['local_sverl', 'shapley_on_policy', 'shapley_on_value']
    characteristics = explainer.compute_characteristics(
        characteristic_modes, num_rolls=1e5, multi_process=False)
    results = explainer.run_values(characteristics, methods=('shapley', 'banzhaf', 'nucleolus'), normalized=True)

    for method, method_results in results.items():
        for name, values in method_results.items():
            print(f"\n=== {method.capitalize()} ({name}) ===")
            print(values)
            with open(f'{method}_{name}.pkl', 'wb') as file:
                pickle.dump(values, file)
