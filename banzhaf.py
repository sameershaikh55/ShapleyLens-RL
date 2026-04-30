import numpy as np
from collections import defaultdict
from utils import F_not_i


class Banzhaf:
    """
    Calculates Banzhaf values given characteristic values.
    """

    def __init__(self, states_to_explain):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain

    def run(self, characteristic_values, normalized=True):
        """
        Calculates all Banzhaf values for every state and feature.
        If normalized=True, divides by 2^(n-1); otherwise, returns raw sum.
        """
        banzhaf_values = defaultdict(lambda: [[] for _ in range(self.F_card)])
        normalizer = 2 ** (self.F_card - 1) if normalized else 1

        for state in self.states:
            C_values = {C: value_table[tuple(state)] for C, value_table in characteristic_values.items()}

            for feature in self.F:
                for C in F_not_i(self.F, feature):  # All coalitions without feature
                    C_with_i = np.append(C, feature).astype(int)
                    C_with_i.sort()
                    banzhaf_values[tuple(state)][feature].append(
                        C_values[tuple(C_with_i)] - C_values[tuple(C)]
                    )

                banzhaf_values[tuple(state)][feature] = (
                    np.sum(banzhaf_values[tuple(state)][feature], axis=0) / normalizer
                )

        return dict(banzhaf_values)
