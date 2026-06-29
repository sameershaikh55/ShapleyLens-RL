import numpy as np
from collections import defaultdict
from utils import F_not_i


class Banzhaf:
    """
    Calculates Banzhaf values given characteristic values.
    """

    def __init__(self, states_to_explain, normalized=False, scale_factor=1.0):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain
        self.normalized = normalized
        self.scale_factor = float(scale_factor)

    def run(self, characteristic_values):
        """
        Calculates all Banzhaf values for every state and feature.
        If normalized=True, divides by 2^(n-1); otherwise, returns raw sum.
        The result is multiplied by scale_factor in both modes.
        """
        banzhaf_values = defaultdict(lambda: [[] for _ in range(self.F_card)])
        normalizer = 2 ** (self.F_card - 1) if self.normalized else 1

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
                ) * self.scale_factor

        return dict(banzhaf_values)

    def run_all(self, characteristic_values_list, labels=None, normalized=True):
        """
        Run Banzhaf for multiple characteristic value sets in one call.

        Args:
            characteristic_values_list: list of characteristic-value dictionaries.
            labels: optional list of names for each characteristic set.
            normalized: whether to normalize each Banzhaf result.
        Returns:
            dict mapping label -> Banzhaf values dict.
        """
        if labels is None:
            labels = [f'set_{i}' for i in range(len(characteristic_values_list))]
        if len(labels) != len(characteristic_values_list):
            raise ValueError('labels must have same length as characteristic_values_list')

        return {
            label: self.run(characteristic_values, normalized=normalized)
            for label, characteristic_values in zip(labels, characteristic_values_list)
        }
