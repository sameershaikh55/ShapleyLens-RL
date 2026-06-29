import math
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from utils import F_not_i


class Shapley:
    """
    Calculates Shapley values given characteristic values.
    """

    def __init__(self, states_to_explain):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain

    def run(self, characteristic_values):
        """
        Calculates all the shapley values for every state and feature.
        """

        shapley_values = defaultdict(lambda: [[] for _ in range(self.F_card)])

        for state in self.states:

            # All characteristic values for a given state.
            C_values = {C: value_table[tuple(state)] for C, value_table in characteristic_values.items()}
            
            for feature in self.F:

                for C in F_not_i(self.F, feature): # All coalitions without feature

                    # Cardinal of C
                    C_card = len(C)

                    # Add our feature to the current coalition
                    C_with_i = np.append(C, feature).astype(int)
                    C_with_i.sort()

                    # Rolling sum, following formula
                    shapley_values[tuple(state)][feature].append(math.factorial(C_card) * math.factorial(self.F_card - C_card - 1) * (C_values[tuple(C_with_i)] - C_values[tuple(C)]))

                # Final weighting and return
                shapley_values[tuple(state)][feature] = np.sum(shapley_values[tuple(state)][feature], axis=0) / math.factorial(self.F_card)

        return dict(shapley_values)

    def run_monte_carlo(self, characteristic_fn, num_samples: int) -> dict:
        """
        Monte-Carlo-Shapley via Permutations-Sampling.

        Approximiert Shapley-Werte ohne vollständige 2^n Koalitionsberechnung.
        Für jede zufällige Permutation der Features wird der marginale Beitrag
        jedes Features entlang der Permutation berechnet.

        Args:
            characteristic_fn: Callable(C: tuple) → {state_tuple: scalar}.
                               On-demand berechnet; Ergebnisse werden intern gecacht.
            num_samples:       Anzahl zufälliger Permutationen.

        Returns:
            {state_tuple: [shapley_value_feature_0, ...]}  — gleiche Struktur wie run().
        """
        cache: dict = {}

        def cv(C: tuple, state_key: tuple):
            if C not in cache:
                cache[C] = characteristic_fn(C)
            return cache[C][state_key]

        shapley_acc = {
            tuple(state): np.zeros(self.F_card, dtype=float)
            for state in self.states
        }

        rng = np.random.default_rng()
        F_list = [int(f) for f in self.F]

        pbar = tqdm(range(num_samples), desc="MC-Shapley", ncols=100)
        for _ in pbar:
            perm = rng.permutation(F_list)
            for state in self.states:
                state_key = tuple(state)
                for k in range(len(perm)):
                    feature = int(perm[k])
                    C_before = tuple(sorted(int(perm[j]) for j in range(k)))
                    C_after = tuple(sorted(int(perm[j]) for j in range(k + 1)))
                    shapley_acc[state_key][feature] += cv(C_after, state_key) - cv(C_before, state_key)
            pbar.set_postfix(cache=len(cache))

        return {
            state_key: list(acc / num_samples)
            for state_key, acc in shapley_acc.items()
        }
