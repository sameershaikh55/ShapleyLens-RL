import numpy as np

class Gately:
    """
    Berechnet den Gately-Punkt basierend auf den charakteristischen Werten.
    Der Gately-Punkt minimiert die maximale 'Propensity to Disrupt'.
    """

    def __init__(self, states_to_explain, normalized=True):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain
        self.normalized = normalized

    def _compute(self, v_N, v_single, v_N_minus_i):
        """
        Berechnet die Gately-Anteile für einen skalaren (ggf. 0-normalisierten) Spielwert.
        Formel: x_i = v({i}) + [(v(N) - v(N\\{i}) - v({i})) / Summe_Zähler] * (v(N) - Summe_v_single)
        """
        numerators = []
        for i in range(self.F_card):
            num = v_N - v_N_minus_i[i] - v_single[i]
            numerators.append(num)

        total_numerator = np.sum(numerators)
        total_v_single = np.sum(v_single)

        if abs(total_numerator) < 1e-12:
            share = v_N / self.F_card
            return [share] * self.F_card

        return [
            v_single[i] + (numerators[i] / total_numerator) * (v_N - total_v_single)
            for i in range(self.F_card)
        ]

    def run(self, characteristic_values):
        """
        Berechnet den Gately-Punkt für jeden Zustand.

        Wenn normalized=True, werden alle Koalitionswerte um v(∅) verschoben,
        sodass ∑ x_i = v(N) − v(∅) (gleiche Baseline wie Shapley/Tau/Nucleolus).
        """
        gately_values = {}
        grand_C = tuple(self.F)
        empty_C = ()

        for state in self.states:
            state_tuple = tuple(state)

            C_values = {
                C: vt[state_tuple]
                for C, vt in characteristic_values.items()
            }
            v_N = C_values[grand_C]

            if isinstance(v_N, np.ndarray) and v_N.ndim >= 1:
                n_components = v_N.size
                per_comp = []
                for k in range(n_components):
                    scalar_C_values = {
                        C: (float(val.flat[k]) if isinstance(val, np.ndarray) else float(val))
                        for C, val in C_values.items()
                    }
                    if self.normalized:
                        v_empty_k = scalar_C_values.get(empty_C, 0.0)
                        scalar_C_values = {C: v - v_empty_k for C, v in scalar_C_values.items()}
                    v_N_k = scalar_C_values[grand_C]
                    v_single = [scalar_C_values[tuple([i])] for i in self.F]
                    v_N_minus_i = [
                        scalar_C_values[tuple(f for f in self.F if f != i)]
                        for i in self.F
                    ]
                    per_comp.append(self._compute(v_N_k, v_single, v_N_minus_i))
                gately_values[state_tuple] = np.array(per_comp).T
            else:
                scalar_C_values = {C: float(v) for C, v in C_values.items()}
                if self.normalized:
                    v_empty = scalar_C_values.get(empty_C, 0.0)
                    scalar_C_values = {C: v - v_empty for C, v in scalar_C_values.items()}
                v_N_s = scalar_C_values[grand_C]
                v_single = [scalar_C_values[tuple([i])] for i in self.F]
                v_N_minus_i = [
                    scalar_C_values[tuple(f for f in self.F if f != i)]
                    for i in self.F
                ]
                gately_values[state_tuple] = self._compute(v_N_s, v_single, v_N_minus_i)

        return gately_values
