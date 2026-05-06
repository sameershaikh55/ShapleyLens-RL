"""
tic_tac_toe/run.py
==================
Runs the Tic-Tac-Toe environment end-to-end:
  1. Train a Q-learning agent
  2. Compute three SVERL characteristic functions
  3. Compute Shapley values  (classic SVERL)
  4. Compute Nucleolus values (cooperative game theory solution concept)
  5. Print a side-by-side summary with an efficiency-axiom verification

TTT state: 9-cell board (0=empty, 1=X, 2=O) → 9 features.
For n=9 features, Shapley ≠ Nucleolus in general (they coincide only
for n=2 after 0-normalisation).

Multiprocessing note
--------------------
The `if __name__ == '__main__':` guard below is REQUIRED on Windows.
"""

import sys
sys.path.insert(0, '../')

import pickle
import numpy as np

from q_agent_2       import Agent
from tic_tac_toe     import TTT
from utils           import train, get_state_dist, F_not_i, tqdm_label
from characteristics import Characteristics
from shapley         import Shapley
from nucleolus       import Nucleolus   # Nucleolus: lexicographic min-max excess


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 1. Environment + agent setup
    # ------------------------------------------------------------------
    env   = TTT()
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=1, alpha=0.2)

    # One mid-game board state to explain
    states_to_explain = np.array([[0, 0, 0,
                                   0, 1, 0,
                                   2, 0, 2]]).astype(int)

    # ------------------------------------------------------------------
    # 2. Train the Q-learning agent
    # ------------------------------------------------------------------
    train(agent, env, 1e5)

    # ------------------------------------------------------------------
    # 3. Derive policy and value table
    #    valid_dict restricts actions to legal board moves only
    # ------------------------------------------------------------------
    agent.get_policy(env.valid_dict)
    agent.get_value_table(env.valid_dict)

    # ------------------------------------------------------------------
    # 4. Approximate the limiting state distribution via rollouts
    # ------------------------------------------------------------------
    state_dist = get_state_dist(agent, env, 1e5)

    # ------------------------------------------------------------------
    # 5. Compute pi_C and v_C for every coalition C
    #    valid_dict ensures only legal actions are sampled
    # ------------------------------------------------------------------
    F_all = np.arange(env.state_dim)

    pi_Cs = {
        tuple(C): agent.get_pi_C(C, state_dist, states_to_explain, env.valid_dict)
        for C in tqdm_label(F_not_i(F_all), 'Calculating all pi_C')
    }

    v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states_to_explain)
        for C in tqdm_label(F_not_i(F_all), 'Calculating all v_C')
    }

    # ------------------------------------------------------------------
    # 6. Compute three characteristic functions
    #    fast_local_sverl is used because TTT's value function is
    #    deterministic (same value for every visit to a state), making
    #    Monte-Carlo rollouts redundant.
    # ------------------------------------------------------------------
    characteristics = Characteristics(env, states_to_explain)

    local_char  = characteristics.fast_local_sverl_C_values(
                      pi_Cs=pi_Cs,
                      multi_process=True, num_p=8)

    policy_char = characteristics.shapley_on_policy(
                      pi_Cs=pi_Cs,
                      multi_process=True, num_p=8)

    value_char  = characteristics.shapley_on_value(
                      v_Cs=v_Cs,
                      multi_process=True, num_p=8)

    char_list = [local_char, policy_char, value_char]
    name_list = ['local', 'policy', 'value_function']

    # ------------------------------------------------------------------
    # 7. Shapley values — compute, pickle, and collect results
    #    Files: local.pkl, policy.pkl, value_function.pkl
    # ------------------------------------------------------------------
    shapley = Shapley(states_to_explain)
    shapley_results = {}

    for char_vals, fname in zip(char_list, name_list):
        sv = shapley.run(char_vals)
        shapley_results[fname] = sv
        with open(f'{fname}.pkl', 'wb') as f:
            pickle.dump(sv, f)

    # ------------------------------------------------------------------
    # 8. Nucleolus values — compute, pickle, and collect results
    #    Files: nucleolus_local.pkl, etc.
    #    For n=9 features (TTT), Shapley ≠ Nucleolus in general.
    #    The efficiency axiom is still satisfied by construction.
    # ------------------------------------------------------------------
    nucleolus = Nucleolus(states_to_explain)
    nucleolus_results = {}

    for char_vals, fname in zip(char_list, name_list):
        nuc_vals = nucleolus.run(char_vals)
        nucleolus_results[fname] = (nuc_vals, char_vals)
        with open(f'nucleolus_{fname}.pkl', 'wb') as f:
            pickle.dump(nuc_vals, f)

    # ------------------------------------------------------------------
    # 9. Presentation-ready summary
    #    Change FOCAL to inspect a different state.
    # ------------------------------------------------------------------
    FOCAL = tuple(states_to_explain[0])   # (0,0,0,0,1,0,2,0,2)

    def _fmt(arr):
        """Format a 1-D array as '[ v0 , v1 , ... ]' with 4 sig-figs."""
        return "[ " + " , ".join(f"{v:.4g}" for v in np.asarray(arr).flatten()) + " ]"

    grand_C = tuple(np.arange(env.state_dim))
    empty_C = ()

    char_vf    = nucleolus_results['value_function'][1]
    nuc_vf     = nucleolus_results['value_function'][0][FOCAL]
    nuc_vN     = float(char_vf[grand_C][FOCAL])
    nuc_vempty = float(char_vf.get(empty_C, {}).get(FOCAL, 0.0))
    nuc_target = nuc_vN - nuc_vempty
    nuc_sum    = float(np.sum(nuc_vf))
    nuc_err    = abs(nuc_sum - nuc_target)
    nuc_flag   = "OK" if nuc_err < 1e-5 else "FAIL"

    shap_vf  = shapley_results['value_function'][FOCAL]
    shap_sum = float(np.sum(shap_vf))

    print("\n" + "=" * 60)
    print(f"Feature Importance: State {FOCAL}")
    print("=" * 60)
    print()
    print("Shapley Values (value_function characteristic):")
    print(f"   State {FOCAL}: {_fmt(shap_vf)}  |  sum = {shap_sum:+.6f}")
    print()
    print("Nucleolus Values (value_function characteristic):")
    print(f"   State {FOCAL}: {_fmt(nuc_vf)}  |  sum = {nuc_sum:+.6f}")
    print()
    print("Efficiency Axiom Verification [Nucleolus]:")
    print(
        f"   sum(x_i)={nuc_sum:+.6f}  "
        f"v(N)-v(\u2205)={nuc_target:+.6f}  "
        f"error={nuc_err:.2e}  [{nuc_flag}]"
    )
    print("=" * 60)