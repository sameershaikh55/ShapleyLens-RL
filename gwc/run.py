import sys
sys.path.insert(0, '../')

import pickle
import numpy as np
from q_agent_1 import Agent
from gwc import Grid
from utils import train
from explainer import Explainer

if __name__ == '__main__':
    env = Grid()
    agent = Agent(env.state_dim, env.num_actions, epsilon=1, gamma=1, alpha=0.2)
    states_to_explain = np.array([[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]])

    # ------------------------------------------------- TRAIN
    train(agent, env, 1e7)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy()

    # ------------------------------------------------- EXPLAINER
    explainer = Explainer(env, agent, states_to_explain)
    explainer.compute_state_dist(sample_size=1e7)
    explainer.compute_pi_Cs()
    explainer.compute_v_Cs()
    characteristic_modes = ['local_sverl', 'global_sverl', 'shapley_on_policy', 'shapley_on_value']
    characteristics = explainer.compute_characteristics(
        characteristic_modes, num_rolls=1e6, multi_process=True, num_p=5)
    results = explainer.run_values(characteristics, methods=('shapley', 'banzhaf'), normalized=True)

    for method, method_results in results.items():
        for name, values in method_results.items():
            print(f"\n=== {method.capitalize()} ({name}) ===")
            print(values)
            with open(f'{method}_{name}.pkl', 'wb') as file:
                pickle.dump(values, file)
