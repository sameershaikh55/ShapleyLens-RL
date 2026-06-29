import numpy as np


class UtopiaPayoff:
    def __init__(self, states_to_explain, normalized=False, scale_factor=1.0):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain
        self.normalized = normalized
        self.scale_factor = float(scale_factor)

    def _normalize_scalar_values(self, feature_values, v_full, v_empty):
        values = np.asarray(feature_values, dtype=float)
        total = np.asarray(v_full).item() - np.asarray(v_empty).item()
        s = values.sum()

        if np.isclose(s, 0.0):
            return feature_values

        normalized = values * (total / s)
        return list(normalized * self.scale_factor)

    def _scale_values(self, feature_values):
        values = np.asarray(feature_values, dtype=float)
        if np.isclose(self.scale_factor, 1.0):
            return list(values)
        return list(values * self.scale_factor)

    def run(self, characteristic_values):
        utopia_values = {}

        full_coalition = tuple(self.F)
        empty_coalition = tuple([])

        for state in self.states:
            state_key = tuple(state)

            C_values = {C: value_table[state_key] for C, value_table in characteristic_values.items()}

            v_full = np.asarray(C_values[full_coalition])

            feature_values = []
            # vectorized version
            for feature in self.F:
                coalition_without_i = tuple(self.F[self.F != feature])
                v_without_i = np.asarray(C_values[coalition_without_i])

                feature_values.append(v_full - v_without_i)

            # Optionaly: Normalize values
            if self.normalized:
                if np.asarray(feature_values[0]).shape == ():
                    if empty_coalition in C_values:
                        feature_values = self._normalize_scalar_values(
                            feature_values, v_full, C_values[empty_coalition]
                        )
            elif not np.isclose(self.scale_factor, 1.0):
                feature_values = self._scale_values(feature_values)

            utopia_values[state_key] = feature_values

        return utopia_values