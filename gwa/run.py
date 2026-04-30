"""
gwa/run.py
==========
Runs the GWA (2x3 Grid World A) environment end-to-end:
  1. Train a Q-learning agent
  2. Compute all four SVERL characteristic functions
  3. Compute Shapley values  (classic SVERL)
  4. Compute Nucleolus values (cooperative game theory solution concept)
  5. Print a side-by-side summary with an efficiency-axiom verification

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

from q_agent_1       import Agent
from gwa             import Grid
from utils           import train, get_state_dist, F_not_i, tqdm_label
from characteristics import Characteristics
from shapley         import Shapley
from nucleolus       import Nucleolus   # Nucleolus: lexicographic min-max excess


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 1. Environment + agent setup
    # ------------------------------------------------------------------
    env   = Grid()
    agent = Agent(env.state_dim, env.num_actions, epsilon=1, gamma=1, alpha=0.2)

    # All reachable non-terminal states
    states_to_explain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # ------------------------------------------------------------------
    # 2. Train the Q-learning agent
    # ------------------------------------------------------------------
    train(agent, env, 1e5)

    # ------------------------------------------------------------------
    # 3. Derive policy and value table from the learned Q-table
    # ------------------------------------------------------------------
    agent.get_policy()
    agent.get_value_table()

    # ------------------------------------------------------------------
    # 4. Approximate the limiting state distribution via rollouts
    # ------------------------------------------------------------------
    state_dist = get_state_dist(agent, env, 1e5)

    # ------------------------------------------------------------------
    # 5. Compute pi_C and v_C for every coalition C
    #    pi_C : marginalised policy when only features in C are observed
    #    v_C  : coalition value (expected return) under pi_C
    # ------------------------------------------------------------------
    F_all = np.arange(env.state_dim)

    pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states_to_explain))
        for C in tqdm_label(F_not_i(F_all), 'Calculating all pi_C')
    }

    v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states_to_explain)
        for C in tqdm_label(F_not_i(F_all), 'Calculating all v_C')
    }

    # ------------------------------------------------------------------
    # 6. Compute all four characteristic functions
    #    - local_sverl  : local SVERL (state-conditioned rollouts)
    #    - global_sverl : global SVERL (distribution-weighted rollouts)
    #    - policy_char  : Shapley-on-policy (KL-divergence based)
    #    - value_char   : Shapley-on-value  (value-function based)
    #
    #    num_rolls=5e3 gives good accuracy; reduce if RAM/time is tight.
    #    Reduce num_p to match your CPU core count if needed.
    # ------------------------------------------------------------------
    characteristics = Characteristics(env, states_to_explain)

    local_char  = characteristics.local_sverl_C_values(
                      num_rolls=5e3, pi_Cs=pi_Cs,
                      multi_process=True, num_p=8)

    global_char = characteristics.global_sverl_C_values(
                      num_rolls=5e3, pi_Cs=pi_Cs,
                      multi_process=True, num_p=8)

    policy_char = characteristics.shapley_on_policy(
                      pi_Cs=pi_Cs,
                      multi_process=True, num_p=8)

    value_char  = characteristics.shapley_on_value(
                      v_Cs=v_Cs,
                      multi_process=True, num_p=8)

    # Bundle into parallel lists for the loops below
    char_list = [local_char, global_char, policy_char, value_char]
    name_list = ['local', 'global', 'policy', 'value_function']

    # ------------------------------------------------------------------
    # 7. Shapley values — compute, pickle, and collect results
    #    Files: local.pkl, global.pkl, policy.pkl, value_function.pkl
    #    (same filenames as before for backward compatibility)
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
    #    Files: nucleolus_local.pkl, nucleolus_global.pkl, etc.
    #    Uses Maschler's sequential LP algorithm (see nucleolus.py).
    # ------------------------------------------------------------------
    nucleolus = Nucleolus(states_to_explain)
    nucleolus_results = {}

    for char_vals, fname in zip(char_list, name_list):
        nuc_vals = nucleolus.run(char_vals)
        # Store (values, char_vals) so the summary can access v(N) below
        nucleolus_results[fname] = (nuc_vals, char_vals)
        with open(f'nucleolus_{fname}.pkl', 'wb') as f:
            pickle.dump(nuc_vals, f)

    # ------------------------------------------------------------------
    # 9. Change FOCAL to inspect a different state.
    # ------------------------------------------------------------------
    FOCAL = (0, 0)

    def _fmt(arr):
        """Format a 1-D array as '[ v0 , v1 , ... ]' with 4 sig-figs."""
        return "[ " + " , ".join(f"{v:.4g}" for v in np.asarray(arr).flatten()) + " ]"

    # Grand coalition tuple, e.g. (0, 1) for a 2-feature game
    grand_C = tuple(np.arange(env.state_dim))
    empty_C = ()

    # After 0-normalisation in the Nucleolus, the efficiency axiom becomes:
    #   sum(x_i) == v(N) - v(∅)     <-- same as Shapley's efficiency
    char_vf   = nucleolus_results['value_function'][1]
    nuc_vf    = nucleolus_results['value_function'][0][FOCAL]
    nuc_vN    = float(char_vf[grand_C][FOCAL])
    nuc_vempty = float(char_vf.get(empty_C, {}).get(FOCAL, 0.0))
    nuc_target = nuc_vN - nuc_vempty   # what sum(x_i) should equal
    nuc_sum   = float(np.sum(nuc_vf))
    nuc_err   = abs(nuc_sum - nuc_target)
    nuc_flag  = "OK" if nuc_err < 1e-5 else "FAIL"

    shap_vf   = shapley_results['value_function'][FOCAL]
    shap_sum  = float(np.sum(shap_vf))

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
        f"v(N)-v(∅)={nuc_target:+.6f}  "
        f"error={nuc_err:.2e}  [{nuc_flag}]"
    )
    print("=" * 60)
