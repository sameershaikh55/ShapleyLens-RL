import sys
sys.path.insert(0, '../')
from q_agent_2 import Agent
from tic_tac_toe import TTT
from utils import train, get_state_dist, F_not_i, tqdm_label
from characteristics import Characteristics
import numpy as np

from explainer import Explainer

if __name__ == "__main__":
    explainer = Explainer()

    env = TTT()
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=1, alpha=0.2)
    states_to_explain = np.array([[0, 0, 0,
                                0, 1, 0,
                                2, 0, 2]]).astype(int)
    if explainer.args.cache:
        try:
            local_sverl_characteristics, policy_characteristics, value_characteristics = explainer.get_cache()
        except:
            print("Cache files not found. Please run without --cached True first")
            sys.exit(1)
    else:
        # ------------------------------------------------- TRAIN
        train(agent, env, 1e5)

        # ------------------------------------------------- GET AGENT'S POLICY
        agent.get_policy(env.valid_dict)

        # ------------------------------------------------- APPROXIMATE STATE DIST
        state_dist = get_state_dist(agent, env, 1e5)

        # ------------------------------------------------- GET AGENT'S VALUE TABLE (for SHAP)
        agent.get_value_table(env.valid_dict)

        # ------------------------------------------------- ALL PI_C
        pi_Cs = {tuple(C): agent.get_pi_C(C, state_dist, states_to_explain, env.valid_dict) for C in tqdm_label(F_not_i(np.arange(env.state_dim)), 'Calculating all pi_C')}

        # ------------------------------------------------- ALL V_C
        v_Cs = {tuple(C): agent.get_v_C(C, state_dist, states_to_explain) for C in tqdm_label(F_not_i(np.arange(env.state_dim)), 'Calculating all v_C')}

        # ------------------------------------------------- ALL CHARACTERISTIC VALUES
        characteristics = Characteristics(env, states_to_explain)
        # Can do fast version because value function value function is the same everywhere (value function is deterministic)
        local_sverl_characteristics = characteristics.fast_local_sverl_C_values(pi_Cs=pi_Cs, multi_process=True, num_p=8)
        policy_characteristics = characteristics.shapley_on_policy(pi_Cs=pi_Cs, multi_process=True, num_p=8)
        value_characteristics = characteristics.shapley_on_value(v_Cs=v_Cs, multi_process=True, num_p=8)

    # ------------------------------------------------- EXPLAINER VALUES
    explainer.set_states(states_to_explain)

    char_list = [
            local_sverl_characteristics,
            policy_characteristics,
            value_characteristics,
    ]
    names = ["local", "policy", "value_function"]

    for char_data, characteristic_type in zip(char_list, names):
        explainer_values = explainer.run(char_data)
        explainer.print(explainer_values, characteristic_type)

        import pickle
        with open('{}.pkl'.format(characteristic_type), 'wb') as file: pickle.dump(explainer_values, file)

        if not explainer.args.cache:
            with open('{}.cache'.format(characteristic_type), 'wb') as file: pickle.dump(char_data, file)