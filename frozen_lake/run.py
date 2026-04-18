"""
run.py
------
FrozenLake SVERL Experiment
Struktur identisch zu Taxi

Ausführen:
    cd frozen_lake
    python run.py
"""

# -------------------------------------------------
# PATH FIX (damit utils, characteristics, shapley gefunden werden)
# -------------------------------------------------
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -------------------------------------------------
# Imports
# -------------------------------------------------
import pickle
import numpy as np
from collections import defaultdict

from frozen_lake_wrap import FrozenLakeWrap, HOLES, GOAL
from utils import value_iteration, F_not_i, tqdm_label, mask_state
from characteristics import Characteristics
from shapley import Shapley

# -------------------------------------------------
# Einstellungen
# -------------------------------------------------
GAMMA = 0.99
NUM_ROLLS = 200

BASE_DIR = os.path.dirname(__file__)

ALL_STATES = list(range(16))
TERMINAL_STATES = HOLES + [GOAL]
EXPLAIN_STATES = [s for s in ALL_STATES if s not in TERMINAL_STATES]


# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------
def save_pickle(obj, name):
    path = os.path.join(BASE_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Gespeichert: {name}")


# -------------------------------------------------
# Partielle Policies
# -------------------------------------------------
def build_partial_policies(env, Q_table):
    F = np.arange(env.state_dim)
    all_C = F_not_i(F)
    all_C.append(list(F))

    pi_Cs, v_Cs = {}, {}

    print("\n[2] Partielle Policies berechnen ...")

    for C in tqdm_label(all_C, "Partial Policies"):
        C = list(C)

        # vollständige Koalition
        if sorted(C) == sorted(list(F)):
            pi, v = {}, {}
            for s in range(env.n_states):
                state = env.decode(s)
                q = np.asarray(Q_table[tuple(state)], dtype=float)
                q_max = np.nanmax(q)
                best = (q == q_max).astype(float)
                if best.sum() == 0:
                    best[:] = 1.0
                pi[tuple(state)] = best / best.sum()
                v[tuple(state)] = q_max
            pi_Cs[tuple(C)] = pi
            v_Cs[tuple(C)] = v
            continue

        # partielle Koalitionen
        pi = defaultdict(lambda: np.ones(env.num_actions) / env.num_actions)
        v = defaultdict(float)
        groups = defaultdict(list)

        for s in range(env.n_states):
            state = env.decode(s)
            masked = mask_state(state, env.state_dim, C)
            groups[tuple(masked)].append(s)

        for states in groups.values():
            q_avg = np.mean(
                [np.asarray(Q_table[tuple(env.decode(s))], dtype=float) for s in states],
                axis=0,
            )
            q_max = np.nanmax(q_avg)
            best = (q_avg == q_max).astype(float)
            if best.sum() == 0:
                best[:] = 1.0
            probs = best / best.sum()

            for s in states:
                state = env.decode(s)
                pi[tuple(state)] = probs
                v[tuple(state)] = q_max

        pi_Cs[tuple(C)] = dict(pi)
        v_Cs[tuple(C)] = dict(v)

    return pi_Cs, v_Cs


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print("=" * 60)
    print("SVERL – FrozenLake (alle States)")
    print("=" * 60)

    # [1] Environment + Value Iteration
    env = FrozenLakeWrap(is_slippery=True)
    Q_table, _ = value_iteration(env, gamma=GAMMA)

    states_to_explain = [env.decode(s) for s in EXPLAIN_STATES]

    # [2] Partielle Policies
    pi_Cs, v_Cs = build_partial_policies(env, Q_table)

    # [3] Charakteristische Werte
    char = Characteristics(env, states_to_explain)

    local_C = char.fast_local_sverl_C_values(pi_Cs, num_rolls=NUM_ROLLS)
    policy_C = char.shapley_on_policy(pi_Cs)
    value_C = char.shapley_on_value(v_Cs)

    save_pickle(local_C, "local.pkl")
    save_pickle(policy_C, "policy.pkl")
    save_pickle(value_C, "value_function.pkl")

    # [4] Shapley-Werte
    shap = Shapley(states_to_explain)

    sv_local = shap.run(local_C)
    sv_value = shap.run(value_C)

    # -------------------------------------------------
    # ✅ AUSGABE: Shapley-Werte für ALLE States
    # -------------------------------------------------
    print("\n=== Shapley Values (Local SVERL, alle States) ===")

    for state_int in EXPLAIN_STATES:
        state = tuple(env.decode(state_int))

        if state not in sv_local:
            continue

        sv = np.array(sv_local[state], dtype=float)

        # falls action-wise → Feature-wise mitteln
        if sv.ndim > 1:
            sv = sv.mean(axis=0)

        print(f"\nState {state}:")
        for i, val in enumerate(sv):
            print(f"  Feature {i} ({env.feature_names()[i]}): {val:.4f}")

    print("\n✅ Fertig! Alle FrozenLake-Shapley-Werte wurden ausgegeben.")
    env.close()


if __name__ == "__main__":
    main()