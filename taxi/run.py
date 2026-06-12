import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_agent_1 import Agent
from taxi_wrap import FactoredState
from utils import get_state_dist, F_not_i, find_states_taxi, value_iteration, tqdm_label
from explainer import Explainer
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

    # ------------------------------------------------- EXPLAINER
    explainer = Explainer(env, agent, states_to_explain, instances=instances)
    explainer.state_dist = state_dist
    explainer.compute_pi_Cs()
    explainer.compute_v_Cs()
    characteristic_modes = ['local_sverl', 'shapley_on_policy', 'shapley_on_value']
    characteristic_values = explainer.compute_characteristics(
        characteristic_modes,
        num_rolls=100,
        multi_process=False,
        num_p=17,
    )

    local_sverl_characteristics = characteristic_values['local_sverl']
    shapley_on_policy_characteristics = characteristic_values['shapley_on_policy']
    shapley_on_value_characteristics = characteristic_values['shapley_on_value']

    result={}
    # ------------------------------------------------- SHAPLEY VALUES
    shapley = Shapley(states_to_explain)
<<<<<<< HEAD
    for characteristics, filename in zip([local_sverl_characteristics, 
                                          shapley_on_policy_characteristics, 
                                          shapley_on_value_characteristics], ['local', 'policy', 'value_function']):
        
        shapley_values = shapley.run(characteristics)
        print_explained_values(shapley_values, f"Shapley ({filename})")

        import pickle
        with open('{}.pkl'.format(filename), 'wb') as file: pickle.dump(shapley_values, file)

    # ------------------------------------------------- BANZHAF VALUES
    banzhaf_results = explainer.run_values(
        {
            'local': local_sverl_characteristics,
            'policy': shapley_on_policy_characteristics,
            'value_function': shapley_on_value_characteristics,
        },
        methods=('banzhaf',),
        normalized=True,
    )['banzhaf']
    for filename, banzhaf_values in banzhaf_results.items():
        print_explained_values(banzhaf_values, f"Banzhaf normalized ({filename})")

        import pickle
        with open(f'banzhaf_{filename}.pkl', 'wb') as file: pickle.dump(banzhaf_values, file)

    # ------------------------------------------------- BANZHAF VALUES (UNNORMALIZED)
    banzhaf_results_unnorm = explainer.run_values(
        {
            'local': local_sverl_characteristics,
            'policy': shapley_on_policy_characteristics,
            'value_function': shapley_on_value_characteristics,
        },
        methods=('banzhaf',),
        normalized=False,
    )['banzhaf']
    for filename, banzhaf_values_unnorm in banzhaf_results_unnorm.items():
        print_explained_values(banzhaf_values_unnorm, f"Banzhaf unnormalized ({filename})")

        import pickle
        with open(f'banzhaf_{filename}_unnormalized.pkl', 'wb') as file: pickle.dump(banzhaf_values_unnorm, file)

    # ------------------------------------------------- NUCLEOLUS VALUES
    nucleolus_results = explainer.run_values(
        {
            'local': local_sverl_characteristics,
            'policy': shapley_on_policy_characteristics,
            'value_function': shapley_on_value_characteristics,
        },
        methods=('nucleolus',),
    )['nucleolus']
    for filename, nucleolus_values in nucleolus_results.items():
        print_explained_values(nucleolus_values, f"Nucleolus ({filename})")

        import pickle
        with open(f'nucleolus_{filename}.pkl', 'wb') as file: pickle.dump(nucleolus_values, file)
=======
    result["shapley"] = {}
    for characteristics, filename in zip([local_sverl_characteristics, 
                                        shapley_on_policy_characteristics, 
                                        shapley_on_value_characteristics], ['local', 'global', 'policy', 'value_function']):
        
        shapley_values = shapley.run(characteristics)
        result["shapley"][filename] = shapley_values

    
    # ------------------------------------------------- TAU VALUES
    tau = TauValue(states_to_explain)
    result["tau"] = {}
    for characteristics, filename in zip([local_sverl_characteristics, 
                                        shapley_on_policy_characteristics, 
                                        shapley_on_value_characteristics], ['local', 'global', 'policy', 'value_function']):
        
        tau_values = tau.run(characteristics)
        result["tau"][filename] = tau_values


    # ------------------------------------------------- ANALYZE RESULTS
    analyzer = ResultAnalyzer(result)


    analyzer.save_pickle()
    analyzer.save_json()
    analyzer.save_csv()
    analyzer.save_summary_csv()

    analyzer.save_value_comparison_pickle(left_value="tau", right_value="shapley")
    analyzer.save_value_comparison_csv(left_value="tau", right_value="shapley")

    # ------------------------------------------------- VISUALIZE RESULTS
    visualizer = ResultVisualizer(result)

    visualizer.plot_heatmaps()
    visualizer.plot_difference_heatmaps(left_value="tau", right_value="shapley")
    visualizer.plot_value_comparison_bars()
>>>>>>> result
