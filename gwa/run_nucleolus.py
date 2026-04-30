"""
gwa/run_nucleolus.py
====================
Runs the GWA (2x3 Grid World A) environment, trains a Q-learning agent,
computes all four SVERL characteristic functions, then evaluates both
Shapley values and Nucleolus values for each explained state.

Multiprocessing note
--------------------
The `if __name__ == '__main__':` guard below is REQUIRED on Windows.
Python's multiprocessing uses 'spawn' mode there, which re-imports this
module in every child process.  Without the guard the child processes
would recursively attempt to start workers, causing an infinite loop.
"""

import sys
sys.path.insert(0, '../')

import pickle
import numpy as np

from q_agent_1    import Agent
from gwa          import Grid
from utils        import train, get_state_dist, F_not_i, tqdm_label
from characteristics import Characteristics
from shapley      import Shapley
from nucleolus    import Nucleolus      # <-- new import


if __name__ == '__main__':
    env   = Grid()
    agent = Agent(env.state_dim, env.num_actions, epsilon=1, gamma=1, alpha=0.2)

    # Explain every reachable non-terminal state in GWA
    states_to_explain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # ------------------------------------------------------------------ #
    # 2.  Train                                                           #
    # ------------------------------------------------------------------ #
    train(agent, env, 1e5)

    # ------------------------------------------------------------------ #
    # 3.  Derive policy and value table from learned Q-table              #
    # ------------------------------------------------------------------ #
    agent.get_policy()
    agent.get_value_table()

    # ------------------------------------------------------------------ #
    # 4.  Approximate limiting state distribution                         #
    # ------------------------------------------------------------------ #
    state_dist = get_state_dist(agent, env, 1e5)

    # ------------------------------------------------------------------ #
    # 5.  Compute pi_C and v_C for every coalition C                      #
    # ------------------------------------------------------------------ #
    F_all = np.arange(env.state_dim)

    pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states_to_explain))
        for C in tqdm_label(F_not_i(F_all), 'Calculating all pi_C')
    }

    v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states_to_explain)
        for C in tqdm_label(F_not_i(F_all), 'Calculating all v_C')
    }

    # ------------------------------------------------------------------ #
    # 6.  Compute all four characteristic functions                       #
    #     multi_process=True is safe here because we are inside           #
    #     `if __name__ == '__main__':`.                                   #
    #     Reduce num_p if your machine has fewer cores.                   #
    # ------------------------------------------------------------------ #
    characteristics = Characteristics(env, states_to_explain)

    local_char  = characteristics.local_sverl_C_values(
                      num_rolls=1e3, pi_Cs=pi_Cs,
                      multi_process=True, num_p=4)

    global_char = characteristics.global_sverl_C_values(
                      num_rolls=1e3, pi_Cs=pi_Cs,
                      multi_process=True, num_p=4)

    policy_char = characteristics.shapley_on_policy(
                      pi_Cs=pi_Cs,
                      multi_process=True, num_p=4)

    value_char  = characteristics.shapley_on_value(
                      v_Cs=v_Cs,
                      multi_process=True, num_p=4)

    char_list = [local_char, global_char, policy_char, value_char]
    name_list = ['local', 'global', 'policy', 'value_function']

    # ------------------------------------------------------------------ #
    # 7.  Shapley values  (compute + pickle all; no intermediate prints) #
    # ------------------------------------------------------------------ #
    shapley = Shapley(states_to_explain)
    shapley_results = {}
    for char_vals, fname in zip(char_list, name_list):
        sv = shapley.run(char_vals)
        shapley_results[fname] = sv
        with open(f'shapley_{fname}.pkl', 'wb') as f:
            pickle.dump(sv, f)

    # ------------------------------------------------------------------ #
    # 8.  Nucleolus values  (compute + pickle all; no intermediate prints)#
    # ------------------------------------------------------------------ #
    nucleolus = Nucleolus(states_to_explain)
    nucleolus_results = {}
    for char_vals, fname in zip(char_list, name_list):
        nuc_vals = nucleolus.run(char_vals)
        nucleolus_results[fname] = (nuc_vals, char_vals)
        with open(f'nucleolus_{fname}.pkl', 'wb') as f:
            pickle.dump(nuc_vals, f)

    # ------------------------------------------------------------------ #
    # 9.  Presentation-ready summary for a single focal state            #
    # ------------------------------------------------------------------ #
    FOCAL = (0, 0)   # change this to inspect a different state

    grand_C    = tuple(np.arange(env.state_dim))
    nuc_vf     = nucleolus_results['value_function'][0][FOCAL]
    nuc_vf_vN  = nucleolus_results['value_function'][1][grand_C][FOCAL]
    nuc_sum    = float(np.sum(nuc_vf))
    nuc_vN     = float(nuc_vf_vN)
    nuc_err    = abs(nuc_sum - nuc_vN)
    nuc_flag   = "OK" if nuc_err < 1e-5 else "FAIL"

    shap_vf    = shapley_results['value_function'][FOCAL]

    def _fmt(arr):
        return "[ " + " , ".join(f"{v:.4g}" for v in np.asarray(arr).flatten()) + " ]"

    print("\n" + "=" * 60)
    print(f"Feature Importance: State {FOCAL}")
    print("=" * 60)
    print()
    print("Shapley Values:")
    print(f"   State {FOCAL}: {_fmt(shap_vf)}")
    print()
    print("Nucleolus Values:")
    print(f"   State {FOCAL}: {_fmt(nuc_vf)}")
    print()
    print("Efficiency Verification (Efficiency Axiom):")
    print(
        f"   [Nucleolus] sum={nuc_sum:+.6f}  "
        f"v(N)={nuc_vN:+.6f}  "
        f"error={nuc_err:.2e}  [{nuc_flag}]"
    )
    print("=" * 60)
