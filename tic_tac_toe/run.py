import sys
sys.path.insert(0, '../')
from q_agent_2 import Agent
from tic_tac_toe.tic_tac_toe import TTT
from utils import train, get_state_dist, F_not_i, tqdm_label
from characteristics import Characteristics
import numpy as np

from game_computation import GameComputation

def tic_tac_toe_init():
    env = TTT()
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=1, alpha=0.2)
    states_to_explain = np.array([[0, 0, 0,
                                0, 1, 0,
                                2, 0, 2]]).astype(int)
    return env, agent, states_to_explain

def tic_tac_toe_get_characteristic_modes():
    return ['local_sverl', 'shapley_on_policy', 'shapley_on_value']
    
def tic_tac_toe_run(env=None, agent=None, states_to_explain=None):
    # ------------------------------------------------- TRAIN
    train(agent, env, 1e5)

    # ------------------------------------------------- GET AGENT'S POLICY
    agent.get_policy(env.valid_dict)
    agent.get_value_table(env.valid_dict)

    # -------------------------------------------------- GAME COMPUTATION
    game_compute = GameComputation(env, agent, states_to_explain, valid_dict=env.valid_dict)
    game_compute.compute_state_dist(sample_size=1e7)
    game_compute.compute_pi_Cs()
    game_compute.get_value_table_for_shapley()
    game_compute.compute_v_Cs()
    characteristic_modes = tic_tac_toe_get_characteristic_modes()
    # Can do fast version because value function value function is the same everywhere (value function is deterministic)
    characteristics = game_compute.compute_characteristics(characteristic_modes, multi_process=True, num_p=8)
    return characteristics