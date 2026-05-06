import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_agent_2 import Agent
from minesweeper import Minesweeper
from utils import train, get_state_dist, find_states_minesweeper
from explainer import Explainer
import numpy as np
import copy
import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    env = Minesweeper(length=4, height=4, num_mines=2)
    agent = Agent(state_dim=env.state_dim, num_actions=env.num_actions, epsilon=0.05, gamma=0.99, alpha=0.2)

    states_to_explain = np.array([[0,  0,  1, -1,
                                   0,  1,  2, -1,
                                   0,  1, -1, -1,
                                   0,  1,  1,  1],
                                  [0,  0,  1, -1,
                                   0,  1,  2, -1,
                                   0,  1, -1,  2,
                                   0,  1,  1,  1]])

    # ------------------------------------------------- TRAIN
    train(agent, env, 3e7)
    find_states_minesweeper(agent, env, states_to_explain, 1e7)

    # Make sure optimal actions are actually correct for agent's policy.
    true_q_table = copy.deepcopy(agent.Q_table)
    for state, values in agent.Q_table.items():

        agent.Q_table[state][values < 0 ] = -1

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy(env.valid_dict)
    agent.Q_table = copy.deepcopy(true_q_table)

    # ------------------------------------------------- EXPLAINER
    explainer = Explainer(env, agent, states_to_explain, valid_dict=env.valid_dict)
    explainer.compute_state_dist(sample_size=1e7)
    explainer.compute_pi_Cs(valid_dict=env.valid_dict)
    explainer.compute_v_Cs(valid_dict=env.valid_dict)
    characteristic_modes = ['fast_local_sverl', 'shapley_on_policy', 'shapley_on_value']
    characteristics = explainer.compute_characteristics(
        characteristic_modes, num_rolls=1e6, multi_process=True, num_p=50, valid_dict=env.valid_dict)
    results = explainer.run_values(characteristics, methods=('shapley', 'banzhaf', 'nucleolus'), normalized=True)

    for method, method_results in results.items():
        for name, values in method_results.items():
            print(f"\n=== {method.capitalize()} ({name}) ===")
            print(values)
            with open(f'{method}_{name}.pkl', 'wb') as file:
                pickle.dump(values, file)
