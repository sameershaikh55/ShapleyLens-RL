import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from q_agent_1 import Agent
from gwd.gwd import Grid
from utils import train, get_state_dist
from game_computation import GameComputation

def gwd_init():
    env = Grid(10, 10, 20)
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.1, gamma=1, alpha=0.2)
    return env, agent, None

def gwd_get_characteristic_modes():
    return ['local_sverl', 'shapley_on_policy', 'shapley_on_value']

def gwd_run(env=None, agent=None, states_to_explain=None):
    # ------------------------------------------------- TRAIN
    train(agent, env, 1e7)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy()

    # ------------------------------------------------- EXPLAINER
    state_dist = get_state_dist(agent, env, 1e7)
    states_to_explain = np.unique(np.array(list(state_dist)), axis=0)
    gamec = GameComputation(env, agent, states_to_explain)
    gamec.state_dist = state_dist
    gamec.compute_pi_Cs()
    gamec.get_value_table_for_shapley()
    gamec.compute_v_Cs()
    characteristic_modes = gwd_get_characteristic_modes()
    characteristics = gamec.compute_characteristics(
        characteristic_modes, num_rolls=1e5, multi_process=False)
    return characteristics
