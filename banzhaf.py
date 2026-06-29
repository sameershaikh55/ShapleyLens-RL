from collections import defaultdict

import numpy as np
from tqdm import tqdm

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

    def run_monte_carlo(
        self,
        characteristic_fn,
        num_samples: int,
        normalized: bool = None,
    ) -> dict:
        """
        Monte-Carlo-Banzhaf via zufälliger Binär-Masken.

        Approximiert Banzhaf-Werte ohne vollständige 2^n Koalitionsberechnung.
        Pro Sample wird eine zufällige Teilmenge S ⊆ N gezogen; der marginale
        Beitrag von Feature i wird als v(S+{i}) - v(S) (je nach Zugehörigkeit)
        geschätzt. Der Schätzer ist unverzerrt (unbiased) für den normalisierten
        Banzhaf-Wert.

        Args:
            characteristic_fn: Callable(C: tuple) → {state_tuple: scalar}.
                               On-demand berechnet; Ergebnisse werden intern gecacht.
            num_samples:       Anzahl zufälliger Binär-Masken.
            normalized:        True → teilt durch 2^(n-1), False → roh.
                               None → verwendet self.normalized.

        Returns:
            {state_tuple: [banzhaf_value_feature_0, ...]}  — gleiche Struktur wie run().
        """
        if normalized is None:
            normalized = self.normalized

        cache: dict = {}

        def cv(C: tuple, state_key: tuple):
            if C not in cache:
                cache[C] = characteristic_fn(C)
            return cache[C][state_key]

        banzhaf_acc = {
            tuple(state): np.zeros(self.F_card, dtype=float)
            for state in self.states
        }

        rng = np.random.default_rng()
        F_int = [int(f) for f in self.F]
        n = self.F_card

        pbar = tqdm(range(num_samples), desc="MC-Banzhaf", ncols=100)
        for _ in pbar:
            # Zufällige Teilmenge S ⊆ N (jedes Feature mit p=0.5)
            included = frozenset(f for f in F_int if rng.random() < 0.5)
            S_base = tuple(sorted(included))

            for state in self.states:
                state_key = tuple(state)
                # v(S_base) einmal berechnen (per Cache)
                v_base = cv(S_base, state_key)

                for feat in F_int:
                    if feat in included:
                        # Marginal: v(S) - v(S \ {feat})
                        S_minus = tuple(sorted(included - {feat}))
                        marginal = v_base - cv(S_minus, state_key)
                    else:
                        # Marginal: v(S ∪ {feat}) - v(S)
                        S_plus = tuple(sorted(included | {feat}))
                        marginal = cv(S_plus, state_key) - v_base
                    banzhaf_acc[state_key][feat] += marginal
            pbar.set_postfix(cache=len(cache))

        # The binary-mask estimator already converges to the *normalized* Banzhaf
        # value (E[marginal_i] = φ_i), so dividing by num_samples gives the
        # normalized value directly.  For the unnormalized ("raw") version we
        # scale back up by 2^(n-1).
        factor = 1 if normalized else (2 ** (n - 1))
        return {
            state_key: list(acc * factor / num_samples)
            for state_key, acc in banzhaf_acc.items()
        }

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
