import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_agent_1 import Agent
from taxi.taxi_wrap import FactoredState
from utils import get_state_dist, F_not_i, find_states_taxi, value_iteration, tqdm_label
from game_computation import GameComputation
from characteristics import Characteristics
from shapley import Shapley
from banzhaf import Banzhaf
try:
    import gym  # type: ignore[import]
except ImportError:
    import gymnasium as gym
import numpy as np

def format_numeric(value, precision=4):
    try:
        return round(float(value), precision)
    except Exception:
        return value


def format_value(value, precision=4):
    if isinstance(value, dict):
        return {format_state(k): format_value(v, precision) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [format_value(v, precision) for v in list(value)]
    return format_numeric(value, precision)


def format_state(state):
    try:
        return tuple(float(x) for x in state)
    except Exception:
        return state


def print_explained_values(values, label):
    print(f"\n=== {label} ===")
    for state, value in values.items():
        print(f"{format_state(state)} -> {format_value(value)}")


def taxi_init():
    env = FactoredState(gym.make('Taxi-v3'))
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.1, gamma=0.99, alpha=0.2)
    states_to_explain = np.array([[0, 3, 3, 1], [0, 3, 4, 3]]).astype(float)
    return env, agent, states_to_explain

def taxi_get_characteristic_modes():
    return ['local_sverl', 'shapley_on_policy', 'shapley_on_value']

def taxi_run(env=None, agent=None, states_to_explain=None):
    # ------------------------------------------------- TRAIN
    agent.Q_table, agent.policy = value_iteration(env, agent.gamma)

    # ------------------------------------------------- FIND INSTANCES OF ENVIRONMENT (ONLY TAXI)
    instances = find_states_taxi(agent, env, states_to_explain)

    # ------------------------------------------------- APPROXIMATE STATE DIST
    state_dist = get_state_dist(agent, env, 1e5)

    # ------------------------------------------------- GET AGENT'S VALUE TABLE (for SHAP)
    agent.get_value_table()

    # ------------------------------------------------- EXPLAINER
    gamec = GameComputation(env, agent, states_to_explain, instances=instances)
    gamec.state_dist = state_dist
    gamec.compute_pi_Cs()
    gamec.get_value_table_for_shapley()
    gamec.compute_v_Cs()
    characteristic_modes = taxi_get_characteristic_modes()
    characteristics = gamec.compute_characteristics(
        characteristic_modes, num_rolls=100, multi_process=False, num_p=17, )
    return characteristics