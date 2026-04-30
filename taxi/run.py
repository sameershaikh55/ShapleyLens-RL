import sys
sys.path.insert(0, '../')

from q_agent_1 import Agent
from taxi_wrap import FactoredState
from utils import get_state_dist, F_not_i, find_states_taxi, value_iteration, tqdm_label
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


if __name__ == '__main__':
    env = FactoredState(gym.make('Taxi-v3'))
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.1, gamma=0.99, alpha=0.2)
    states_to_explain = np.array([[0, 3, 3, 1], [0, 3, 4, 3]]).astype(float)

    # ------------------------------------------------- TRAIN
    agent.Q_table, agent.policy = value_iteration(env, agent.gamma)

    # ------------------------------------------------- FIND INSTANCES OF ENVIRONMENT (ONLY TAXI)
    instances = find_states_taxi(agent, env, states_to_explain)

    # ------------------------------------------------- APPROXIMATE STATE DIST
    state_dist = get_state_dist(agent, env, 1e5)

    # ------------------------------------------------- GET AGENT'S VALUE TABLE (for SHAP)
    agent.get_value_table()

    # ------------------------------------------------- ALL PI_C
    pi_Cs = {tuple(C): agent.get_pi_C(C, state_dist, states_to_explain) for C in tqdm_label(F_not_i(np.arange(env.state_dim)), 'Calculating all pi_C')}

    # ------------------------------------------------- ALL V_C
    v_Cs = {tuple(C): agent.get_v_C(C, state_dist, states_to_explain) for C in tqdm_label(F_not_i(np.arange(env.state_dim)), 'Calculating all v_C')}

    # ------------------------------------------------- ALL CHARACTERISTIC VALUES
    characteristics = Characteristics(env, states_to_explain, instances=instances)
    local_sverl_characteristics = characteristics.local_sverl_C_values(num_rolls=100, pi_Cs=pi_Cs, multi_process=False, num_p=17)
    shapley_on_policy_characteristics = characteristics.shapley_on_policy(pi_Cs=pi_Cs, multi_process=False, num_p=17)
    shapley_on_value_characteristics = characteristics.shapley_on_value(v_Cs=v_Cs, multi_process=False, num_p=17)

    # ------------------------------------------------- SHAPLEY VALUES
    shapley = Shapley(states_to_explain)
    for characteristics, filename in zip([local_sverl_characteristics, 
                                          shapley_on_policy_characteristics, 
                                          shapley_on_value_characteristics], ['local', 'policy', 'value_function']):
        
        shapley_values = shapley.run(characteristics)
        print_explained_values(shapley_values, f"Shapley ({filename})")

        import pickle
        with open('{}.pkl'.format(filename), 'wb') as file: pickle.dump(shapley_values, file)

    # ------------------------------------------------- BANZHAF VALUES
    banzhaf = Banzhaf(states_to_explain)
    for characteristics, filename in zip([local_sverl_characteristics,
                                          shapley_on_policy_characteristics,
                                          shapley_on_value_characteristics], ['local', 'policy', 'value_function']):

        banzhaf_values = banzhaf.run(characteristics)
        print_explained_values(banzhaf_values, f"Banzhaf normalized ({filename})")

        import pickle
        with open('banzhaf_{}.pkl'.format(filename), 'wb') as file: pickle.dump(banzhaf_values, file)

    # ------------------------------------------------- BANZHAF VALUES (UNNORMALIZED)
    for characteristics, filename in zip([local_sverl_characteristics,
                                          shapley_on_policy_characteristics,
                                          shapley_on_value_characteristics], ['local', 'policy', 'value_function']):

        banzhaf_values_unnorm = banzhaf.run(characteristics, normalized=False)
        print_explained_values(banzhaf_values_unnorm, f"Banzhaf unnormalized ({filename})")

        import pickle
        with open('banzhaf_{}_unnormalized.pkl'.format(filename), 'wb') as file: pickle.dump(banzhaf_values_unnorm, file)
