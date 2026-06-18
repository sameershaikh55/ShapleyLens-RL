import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from q_agent_1 import Agent
from gwa.gwa import Grid
from utils import train
from game_computation import GameComputation

def gwa_init():
    env = Grid()
    agent = Agent(env.state_dim, env.num_actions, epsilon=1, gamma=1, alpha=0.2)
    states_to_explain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Explaining all states.
    return env, agent, states_to_explain

def gwa_get_characteristic_modes():
    return ['local_sverl', 'global_sverl', 'shapley_on_policy', 'shapley_on_value']

def gwa_run(env=None, agent=None, states_to_explain=None):
    # ------------------------------------------------- TRAIN
    train(agent, env, 1e7)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy()

    # ------------------------------------------------- GAME COMPUTATION
    gamec = GameComputation(env, agent, states_to_explain)
    gamec.compute_state_dist(sample_size=1e7)
    gamec.compute_pi_Cs()
    gamec.get_value_table_for_shapley()
    gamec.compute_v_Cs()
    characteristic_modes = gwa_get_characteristic_modes()
    characteristics = gamec.compute_characteristics(characteristic_modes, num_rolls=1e6, multi_process=True, num_p=5)
    return characteristics

if __name__ == '__main__':
    env = Grid()
    agent = Agent(env.state_dim, env.num_actions, epsilon=1, gamma=1, alpha=0.2)
    states_to_explain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Explaining all states.

    # ------------------------------------------------- TRAIN
    train(agent, env, 1e7)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy()

    # ------------------------------------------------- EXPLAINER
    explainer = Explainer(env, agent, states_to_explain)
    explainer.compute_state_dist(sample_size=1e7)
    explainer.compute_pi_Cs()
    explainer.compute_v_Cs()
    characteristic_modes = ['local_sverl', 'shapley_on_policy', 'shapley_on_value'] # global_sverl
    characteristics = explainer.compute_characteristics(
        characteristic_modes, num_rolls=1e6, multi_process=True, num_p=5)
    results = explainer.run_values(characteristics, methods=('shapley', 'banzhaf', 'nucleolus'), normalized=True)

    for method, method_results in results.items():
        for name, values in method_results.items():
            print(f"\n=== {method.capitalize()} ({name}) ===")
            print(values)
            with open(f'{method}_{name}.pkl', 'wb') as file:
                pickle.dump(values, file)
