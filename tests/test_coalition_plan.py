"""
Tests für coalition_plan.py, MC-Shapley und MC-Banzhaf.

Führe aus mit:  python -m pytest tests/ -v
"""

import math
import sys
import time
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from coalition_plan import (
    CoalitionPlan,
    assess_exact_feasibility,
    coalitions_for_method,
    coalitions_for_shapley_banzhaf_exact,
    merge_coalition_sets,
    plan_computation,
)
from shapley import Shapley
from banzhaf import Banzhaf


# ------------------------------------------------------------------ helpers

def _dummy_char_fn(C):
    """Einfache charakteristische Funktion: v(C) = |C| für jeden Zustand."""
    return {(0.0,): float(len(C))}


def _slow_probe_fn(C):
    """Simuliert eine sehr langsame Koalitionsberechnung (für infeasible-Test)."""
    time.sleep(0.05)
    return {(0.0,): float(len(C))}


def _fast_probe_fn(C):
    """Sehr schnelle Koalitionsberechnung (für feasible-Test)."""
    return {(0.0,): float(len(C))}


# ================================================================ coalition helpers

class TestCoalitionsForMethod:
    """coalitions_for_method liefert exakt die erwarteten Mengen."""

    def test_gately_count(self):
        n = 5
        F = np.arange(n)
        result = coalitions_for_method("gately", F)
        # v(N), v({i}), v(N\{i}), v() → 1 + n + n + 1 = 2n+2
        assert len(result) == 2 * n + 2

    def test_gately_contains_required_coalitions(self):
        F = np.arange(4)
        result = coalitions_for_method("gately", F)
        F_tuple = tuple(F.tolist())
        # grand coalition
        assert F_tuple in result
        # empty coalition
        assert () in result
        # all singletons
        for i in F:
            assert (int(i),) in result
        # all N\{i}
        for i in F:
            assert tuple(f for f in F.tolist() if f != i) in result

    def test_utopia_count(self):
        n = 6
        F = np.arange(n)
        result = coalitions_for_method("utopia-payoff", F)
        # v(N), v(N\{i}), v() → 1 + n + 1 = n+2
        assert len(result) == n + 2

    def test_s_utopia_same_coalitions_as_utopia(self):
        F = np.arange(6)
        assert coalitions_for_method("s_utopia-payoff", F) == coalitions_for_method(
            "utopia-payoff", F
        )

    def test_shapley_returns_full_power_set(self):
        n = 4
        F = np.arange(n)
        result = coalitions_for_method("shapley", F)
        assert len(result) == 2 ** n

    def test_banzhaf_returns_full_power_set(self):
        n = 3
        F = np.arange(n)
        result = coalitions_for_method("banzhaf", F)
        assert len(result) == 2 ** n

    def test_gately_coalitions_are_sorted_tuples(self):
        F = np.arange(4)
        result = coalitions_for_method("gately", F)
        for C in result:
            assert C == tuple(sorted(C))


class TestCoalitionsForShapleyBanzhafExact:
    def test_size(self):
        for n in (1, 3, 5):
            F = np.arange(n)
            assert len(coalitions_for_shapley_banzhaf_exact(F)) == 2 ** n

    def test_contains_empty_and_full(self):
        F = np.arange(4)
        result = coalitions_for_shapley_banzhaf_exact(F)
        assert () in result
        assert tuple(F.tolist()) in result


class TestMergeCoalitionSets:
    def test_union(self):
        a = {(), (0,), (1,)}
        b = {(0,), (2,)}
        assert merge_coalition_sets(a, b) == {(), (0,), (1,), (2,)}

    def test_empty(self):
        assert merge_coalition_sets(set(), set()) == set()


# ================================================================ feasibility

class TestAssessExactFeasibility:

    def test_fast_probe_small_n_is_feasible(self):
        """Für n=4 (16 Koalitionen) und schnellem probe → feasible."""
        feasible, reason, projected = assess_exact_feasibility(
            _fast_probe_fn, n=4, bytes_per_coalition=0, num_probe=3
        )
        assert feasible, f"Erwartet feasible, aber: {reason}"
        assert projected >= 0

    def test_slow_probe_is_infeasible(self):
        """Langsamer probe mit großem n → infeasible (Zeit-Check)."""
        # n=20 → 2^20 ≈ 1M Koalitionen, 50ms/Koalition → projected ≈ 14 Stunden
        feasible, reason, projected = assess_exact_feasibility(
            _slow_probe_fn, n=20, bytes_per_coalition=0, num_probe=2
        )
        assert not feasible, f"Erwartet infeasible, aber: {reason}"
        assert projected > 10.0

    def test_memory_check_triggers(self):
        """Sehr viele Bytes pro Koalition → infeasible per Speicher-Check."""
        # Simuliere 2^30 Bytes pro Koalition bei 30 Features → definitiv zu viel
        feasible, reason, projected = assess_exact_feasibility(
            _fast_probe_fn, n=30, bytes_per_coalition=2**30, num_probe=1
        )
        assert not feasible
        assert "Speicher" in reason or "MC" in reason

    def test_returns_reason_string(self):
        feasible, reason, _ = assess_exact_feasibility(
            _fast_probe_fn, n=3, bytes_per_coalition=0, num_probe=3
        )
        assert isinstance(reason, str) and len(reason) > 0


# ================================================================ plan_computation

class TestPlanComputation:

    def test_gately_only_sparse(self):
        """Nur Gately → sparse exakt, kein Feasibility-Check nötig."""
        F = np.arange(6)
        cp = plan_computation(["gately"], F)
        assert cp.modes["gately"] == "exact"
        assert "shapley" not in cp.modes
        # Koalitionen sind sparse (≤ 2n+2)
        assert len(cp.coalitions) <= 2 * len(F) + 2

    def test_shapley_exact_small_n(self):
        """Für kleines n → exakter Pfad gewählt."""
        F = np.arange(4)
        cp = plan_computation(["shapley", "gately"], F, probe_fn=_fast_probe_fn)
        assert cp.feasible_exact
        assert cp.modes["shapley"] == "exact"
        assert cp.modes["gately"] == "exact"
        assert len(cp.coalitions) == 2 ** 4  # full power set

    def test_shapley_approx_slow_probe(self):
        """Langsamer probe → MC-Approximation für Shapley."""
        F = np.arange(20)
        cp = plan_computation(["shapley"], F, probe_fn=_slow_probe_fn)
        assert not cp.feasible_exact
        assert cp.modes["shapley"] == "approximate"

    def test_nucleolus_skipped_when_infeasible(self):
        """Nucleolus wird übersprungen wenn nicht exakt machbar."""
        F = np.arange(20)
        cp = plan_computation(["nucleolus", "shapley"], F, probe_fn=_slow_probe_fn)
        assert cp.modes.get("nucleolus") == "skip"
        assert "nucleolus" in cp.skip

    def test_nucleolus_exact_when_feasible(self):
        """Nucleolus läuft exakt wenn Feasibility-Check besteht."""
        F = np.arange(3)
        cp = plan_computation(["nucleolus"], F, probe_fn=_fast_probe_fn)
        assert cp.modes.get("nucleolus") == "exact"
        assert "nucleolus" not in cp.skip

    def test_no_probe_fn_assumes_exact(self):
        """Ohne probe_fn wird exakter Pfad angenommen."""
        F = np.arange(10)
        cp = plan_computation(["shapley"], F, probe_fn=None)
        assert cp.feasible_exact
        assert cp.modes["shapley"] == "exact"

    def test_coalitions_are_sorted_tuples(self):
        F = np.arange(4)
        cp = plan_computation(["gately", "shapley"], F, probe_fn=_fast_probe_fn)
        for C in cp.coalitions:
            assert isinstance(C, tuple)
            assert C == tuple(sorted(C))


# ================================================================ MC-Shapley accuracy

def _build_char_fn(n):
    """
    Charakteristische Funktion: v(C) = Σ_{i ∈ C} (i+1).
    Exakter Shapley-Wert von Feature i = (i+1).
    """
    def char_fn(C):
        val = float(sum(int(f) + 1 for f in C))
        return {tuple(float(j) for j in range(n)): val}
    return char_fn


class TestMCShapley:

    def test_mc_converges_to_exact(self):
        """MC-Shapley konvergiert auf exakten Shapley-Wert für kleines n."""
        n = 5
        states = [np.array([float(i) for i in range(n)])]
        F = np.arange(n)
        char_fn = _build_char_fn(n)

        # Exaktes Ergebnis
        exact_chars = {
            tuple(int(f) for f in coal): char_fn(coal)
            for coal in coalitions_for_shapley_banzhaf_exact(F)
        }
        shap = Shapley(states)
        exact = shap.run(exact_chars)

        # MC-Ergebnis
        mc = shap.run_monte_carlo(char_fn, num_samples=2000)

        state_key = tuple(float(i) for i in range(n))
        for feat in range(n):
            assert abs(exact[state_key][feat] - mc[state_key][feat]) < 0.15, (
                f"Feature {feat}: exact={exact[state_key][feat]:.3f}, "
                f"mc={mc[state_key][feat]:.3f}"
            )

    def test_mc_efficiency_property(self):
        """MC-Shapley erfüllt annähernd die Effizienz-Eigenschaft: Σ φ_i ≈ v(N) - v({})."""
        n = 4
        states = [np.array([float(i) for i in range(n)])]
        char_fn = _build_char_fn(n)
        shap = Shapley(states)
        mc = shap.run_monte_carlo(char_fn, num_samples=3000)

        state_key = tuple(float(i) for i in range(n))
        total = sum(mc[state_key])
        v_N = float(sum(i + 1 for i in range(n)))
        v_empty = 0.0
        assert abs(total - (v_N - v_empty)) < 0.5, (
            f"Effizienz verletzt: Σφ={total:.3f}, v(N)={v_N}"
        )

    def test_mc_result_has_correct_structure(self):
        # Use one state whose tuple key matches what _build_char_fn returns
        n = 3
        states = [np.array([float(i) for i in range(n)])]
        char_fn = _build_char_fn(n)
        shap = Shapley(states)
        result = shap.run_monte_carlo(char_fn, num_samples=100)

        state_key = tuple(float(i) for i in range(n))
        assert state_key in result
        assert len(result[state_key]) == n


# ================================================================ MC-Banzhaf accuracy

class TestMCBanzhaf:

    def test_mc_converges_to_exact_normalized(self):
        """MC-Banzhaf konvergiert auf exakten normalisierten Banzhaf-Wert."""
        n = 4
        states = [np.array([float(i) for i in range(n)])]
        F = np.arange(n)
        char_fn = _build_char_fn(n)

        exact_chars = {
            tuple(int(f) for f in coal): char_fn(coal)
            for coal in coalitions_for_shapley_banzhaf_exact(F)
        }
        banz = Banzhaf(states, normalized=True)
        exact = banz.run(exact_chars)

        mc = banz.run_monte_carlo(char_fn, num_samples=3000, normalized=True)

        state_key = tuple(float(i) for i in range(n))
        for feat in range(n):
            assert abs(exact[state_key][feat] - mc[state_key][feat]) < 0.2, (
                f"Feature {feat}: exact={exact[state_key][feat]:.3f}, "
                f"mc={mc[state_key][feat]:.3f}"
            )

    def test_mc_result_has_correct_structure(self):
        # Use a state whose tuple key matches what _build_char_fn returns
        n = 3
        states = [np.array([float(i) for i in range(n)])]
        char_fn = _build_char_fn(n)
        banz = Banzhaf(states)
        result = banz.run_monte_carlo(char_fn, num_samples=50)
        state_key = tuple(float(i) for i in range(n))
        assert state_key in result
        assert len(result[state_key]) == n
