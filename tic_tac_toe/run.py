import sys
sys.path.insert(0, '../')

import pickle

from q_agent_2 import Agent
from tic_tac_toe import TTT
from utils import train, get_state_dist
from explainer import Explainer
import numpy as np

if __name__ == '__main__':
    env = TTT()
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=1, alpha=0.2)
    states_to_explain = np.array([[0, 0, 0,
                                   0, 1, 0,
                                   2, 0, 2]]).astype(int)

    # ------------------------------------------------- TRAIN
    train(agent, env, 1e7)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy(env.valid_dict)

    # ------------------------------------------------- EXPLAINER
    explainer = Explainer(env, agent, states_to_explain, valid_dict=env.valid_dict)
    explainer.compute_state_dist(sample_size=1e7)
    explainer.compute_pi_Cs(valid_dict=env.valid_dict)
    explainer.compute_v_Cs(valid_dict=env.valid_dict)
    characteristic_modes = ['fast_local_sverl']
    characteristics = explainer.compute_characteristics(
        characteristic_modes, num_rolls=1, multi_process=False, num_p=1, valid_dict=env.valid_dict)
    results = explainer.run_values(characteristics, methods=('shapley', 'banzhaf'), normalized=True)

    for method, method_results in results.items():
        for name, values in method_results.items():
            print(f"\n=== {method.capitalize()} ({name}) ===")
            print(values)
            with open(f'{method}_{name}.pkl', 'wb') as file:
                pickle.dump(values, file)