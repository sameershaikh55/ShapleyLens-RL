import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_agent_2 import Agent
from minesweeper.minesweeper import Minesweeper
from utils import train, get_state_dist, find_states_minesweeper
from game_computation import GameComputation
import numpy as np
import copy
import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def minesweeper_init():
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
    return env, agent, states_to_explain

def minesweeper_get_characteristic_modes():
    return ['fast_local_sverl', 'shapley_on_policy', 'shapley_on_value']

def minesweeper_run(env=None, agent=None, states_to_explain=None):
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
    gamec = GameComputation(env, agent, states_to_explain, valid_dict=env.valid_dict)
    gamec.compute_state_dist(sample_size=1e7)
    gamec.compute_pi_Cs(valid_dict=env.valid_dict)
    gamec.get_value_table_for_shapley()
    gamec.compute_v_Cs(valid_dict=env.valid_dict)
    characteristic_modes = minesweeper_get_characteristic_modes()
    characteristics = gamec.compute_characteristics(
        characteristic_modes, num_rolls=1e6, multi_process=True, num_p=50, valid_dict=env.valid_dict)
    return characteristics
