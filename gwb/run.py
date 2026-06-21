import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from q_agent_1 import Agent
from gwb.gwb import Grid
from utils import train
from game_computation import GameComputation

def gwb_init():
    env = Grid()
    agent = Agent(env.state_dim, env.num_actions, epsilon=1, gamma=1, alpha=0.2)
    states_to_explain = np.array([[0, 0], [1, 0], [1, 1], [1, 2]])
    return env, agent, states_to_explain

def gwb_get_characteristic_modes():
    return ['local_sverl', 'global_sverl', 'shapley_on_policy', 'shapley_on_value']

def gwb_run(env=None, agent=None, states_to_explain=None):
    # ------------------------------------------------- TRAIN
    train(agent, env, 1e7)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy()

    # ------------------------------------------------- EXPLAINER
    gamec = GameComputation(env, agent, states_to_explain)
    gamec.compute_state_dist(sample_size=1e7)
    gamec.compute_pi_Cs()
    gamec.get_value_table_for_shapley()
    gamec.compute_v_Cs()
    characteristic_modes = gwb_get_characteristic_modes()
    characteristics = gamec.compute_characteristics(
        characteristic_modes, num_rolls=1e6, multi_process=True, num_p=5)
    return characteristics