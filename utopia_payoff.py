import numpy as np

class UtopiaPayoff:
    def __init__(self, states_to_explain):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain

    def run(self, characteristic_values):
        utopia_values = {}

        full_coalition = tuple(self.F)

        for state in self.states:
            state_key = tuple(state)

            C_values = { C: value_table[state_key] for C, value_table in characteristic_values.items() }

            v_full = np.asarray(C_values[full_coalition])

            feature_values = []
            # vectorized version
            for feature in self.F:
                coalition_without_i = tuple(self.F[self.F != feature])
                v_without_i = np.asarray(C_values[coalition_without_i])

                feature_values.append(v_full - v_without_i)

            utopia_values[state_key] = feature_values

        return utopia_values