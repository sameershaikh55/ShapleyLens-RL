import numpy as np
from collections import defaultdict

class Gately:
    """
    Berechnet den Gately-Punkt basierend auf den charakteristischen Werten.
    Der Gately-Punkt minimiert die maximale 'Propensity to Disrupt'.
    """

    def __init__(self, states_to_explain):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain

    def run(self, characteristic_values):
        """
        Berechnet den Gately-Punkt für jeden Zustand.
        """
        gately_values = defaultdict(list)

        for state in self.states:
            state_tuple = tuple(state)
            
            # 1. v(N): Wert der vollen Koalition (alle Features)
            full_coalition = tuple(self.F)
            v_N = characteristic_values[full_coalition][state_tuple]
            
            # 2. v({i}): Einzelwerte für jedes Feature
            v_single = []
            for i in self.F:
                # v({i}) abrufen
                v_single.append(characteristic_values[tuple([i])][state_tuple])
            
            # 3. v(N \ {i}): Wert der Koalition ohne das jeweilige Feature
            v_N_minus_i = []
            for i in self.F:
                # Erstelle Koalition ohne Feature i
                N_minus_i = tuple([f for f in self.F if f != i])
                v_N_minus_i.append(characteristic_values[N_minus_i][state_tuple])

            # 4. Berechnung der Gately-Anteile
            # Formel: x_i = v({i}) + [(v(N) - v(N\{i\}) - v({i})) / Summe_aller_Zähler] * (v(N) - Summe_v_single)
            
            numerators = []
            for i in range(self.F_card):
                # Zähler: Was verliert die Gruppe, wenn i geht, minus was i alleine bringt
                num = v_N - v_N_minus_i[i] - v_single[i]
                numerators.append(num)
            
            total_numerator = np.sum(numerators)
            total_v_single = np.sum(v_single)

            current_state_gately = []
            
            # Falls die Summe der Zähler 0 ist (kein Feature hat Einfluss), 
            # verteilen wir den Wert gleichmäßig, um Division durch Null zu vermeiden.
            if abs(total_numerator) < 1e-12:
                share = v_N / self.F_card
                current_state_gately = [share] * self.F_card
            else:
                for i in range(self.F_card):
                    x_i = v_single[i] + (numerators[i] / total_numerator) * (v_N - total_v_single)
                    current_state_gately.append(x_i)
            
            gately_values[state_tuple] = current_state_gately

        return dict(gately_values)
