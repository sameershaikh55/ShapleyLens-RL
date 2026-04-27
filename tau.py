import numpy as np
from collections import defaultdict
from itertools import combinations


class TauValue:
    """
    Calculates Tau values given characteristic values.

    Expected input format:
        characteristic_values = {
            tuple(C): {
                tuple(state): scalar_value
            }
        }

    Output format:
        {
            tuple(state): [tau_feature_0, tau_feature_1, ...]
        }
    """

    def __init__(self, states_to_explain, strict=True):
        self.states = states_to_explain
        self.F_card = len(states_to_explain[0]) #Dimension of State-Features
        self.F = np.arange(self.F_card) #List with length of f_card
        self.strict = strict

    def _all_coalitions(self):
        """All coalitions C subseteq F."""
        coalitions = []
        for r in range(self.F_card + 1):
            for C in combinations(self.F, r):
                coalitions.append(tuple(sorted(C))) #Tuple of Feature-Coalitions
        return coalitions

    def _coalitions_with_feature(self, feature):
        """
        All possible outsider coalitions containing feature i,
        excluding the grand coalition F.
        """
        F_tuple = tuple(self.F) #TUple of all features / Grand Coalition

        result = []
        for C in self._all_coalitions():
            if feature in C and C != F_tuple:
                result.append(C)

        return result

    def _get_value(self, characteristic_values, C, state):
        """
        Get the characteristic value for a given coalition C and state.
        """
        C = tuple(sorted(C))
        state = tuple(state)

        if C not in characteristic_values:
            raise KeyError(f"Missing characteristic value for coalition {C}")

        if state not in characteristic_values[C]:
            raise KeyError(f"Missing characteristic value for state {state} and coalition {C}")

        return characteristic_values[C][state]

    def _compute_vmax(self, characteristic_values, state):
        """
        v_i,max = v(F) - v(F \\ {i})
        upper bound on the contribution of feature i.
        Contribution of feature i to the grand coalition F.
        """
        F_tuple = tuple(self.F) # Grand Coalition
        grand_value = self._get_value(characteristic_values, F_tuple, state) #characteristic value for the grand coalition

        vmax = np.zeros(self.F_card) #array of zeros with length of number of features

        for i in self.F:
            coalition_without_i = tuple(j for j in self.F if j != i)
            without_i_value = self._get_value(characteristic_values, coalition_without_i, state) #characteristic value for the coalition without feature i
            
            vmax[i] = grand_value - without_i_value

        return vmax

    def _compute_cn(self, characteristic_values, state, feature):
        """
        c_i = v({i})
        Individual threat value of feature i.
        Characteristic value for the coalition containing only feature i.
        """
        return self._get_value(characteristic_values, (feature,), state)

    def _compute_dn(self, characteristic_values, state, feature, vmax):
        """
        d_i = max over outsider coalitions C containing i:
              v(C) - sum_{j in C \\ {i}} v_j,max
        Coalition threat value of feature i.
        Max value of the outsider coalitions C containing feature i minus the sum of upper bounds of features in C excluding feature i.
        """
        best = -np.inf

        for C in self._coalitions_with_feature(feature):
            if len(C) < 2:
                continue

            coalition_value = self._get_value(characteristic_values, C, state)

            compensation = sum(vmax[j] for j in C if j != feature) # Sum of upper bounds for features in the coalition excluding feature i

            threat_value = coalition_value - compensation
            best = max(best, threat_value)

        return best

    def _compute_vmin(self, characteristic_values, state, vmax):
        """
        v_i,min = max(0, c_i, d_i)
        Lower bound on the contribution of feature i.
        """
        vmin = np.zeros(self.F_card)

        for i in self.F:
            cn = self._compute_cn(characteristic_values, state, i)
            dn = self._compute_dn(characteristic_values, state, i, vmax)

            vmin[i] = max(0.0, cn, dn)

        return vmin

    def _integrity_check(self, grand_value, vmax, vmin, state):
        """
        Tau value exists only if:
            sum(vmin) <= v(F) <= sum(vmax)
            vmin_i <= vmax_i for all i
        """
        errors = []

        if np.sum(vmin) > grand_value + 1e-12:
            errors.append(
                f"sum(vmin)={np.sum(vmin)} > grand_value={grand_value}"
            )

        if np.sum(vmax) + 1e-12 < grand_value:
            errors.append(
                f"sum(vmax)={np.sum(vmax)} < grand_value={grand_value}"
            )

        for i in self.F:
            if vmin[i] > vmax[i] + 1e-12:
                errors.append(
                    f"feature {i}: vmin={vmin[i]} > vmax={vmax[i]}"
                )

        if errors and self.strict:
            raise ValueError(
                f"Tau value does not exist for state {tuple(state)}:\n"
                + "\n".join(errors)
            )

        return errors

    def run(self, characteristic_values):
        """
        Calculates Tau values for every state and feature.
        """
        tau_values = defaultdict(lambda: [0.0 for _ in range(self.F_card)])

        F_tuple = tuple(self.F)

        for state in self.states:
            state = tuple(state)

            grand_value = self._get_value(characteristic_values, F_tuple, state)

            v1 = self._compute_vmax(characteristic_values, state)
            v2 = self._compute_vmin(characteristic_values, state, v1)

            vmax=max(v1, v2) #vmax_i = max(v_i,max, v_i,min)
            vmin=min(v1, v2) #vmin_i = min(v_i,max, v_i,min)

            self._integrity_check(grand_value, vmax, vmin, state)

            sum_vmax = np.sum(vmax)
            sum_vmin = np.sum(vmin)

            #gamma = (grand_value - sum_vmin) / (sum_vmax - sum_vmin)
            #if sum_vmax == sum_vmin, we can set gamma to 0.5, since in this case vmin and vmax are the same and any gamma would yield the same tau.
            if abs(sum_vmax - sum_vmin) < 1e-12:
                gamma = 0.5
            # sum((tau*vmax​+(1−tau)vmin​))=v(F) -> Forderung
            # tau*sum(vmax) + (1 - tau)*sum(vmin) = grand_value
            # tau*(sum_vmax - sum_vmin) + sum_vmin = grand_value
            # gamma = (grand_value - sum_vmin) / (sum_vmax - sum_vmin)
            else:
                gamma = (grand_value - sum_vmin) / (sum_vmax - sum_vmin)

            # tau = gamma * (vmax - vmin) + vmin
            # tau = gamma * vmax + (1 - gamma) * vmin
            tau = gamma * vmax + (1 - gamma) * vmin

            tau_values[state] = list(tau)

        return dict(tau_values)