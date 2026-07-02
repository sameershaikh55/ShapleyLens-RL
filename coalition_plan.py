"""
coalition_plan.py — adaptive coalition computation planner.

Determines at runtime (without a fixed feature threshold) whether to use
exact or Monte-Carlo approximation for Shapley/Banzhaf, and which sparse
set of coalitions to precompute for Gately/Utopia.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np


# ------------------------------------------------------------------ constants

_SPARSE_METHODS: frozenset = frozenset(("gately", "utopia-payoff", "s_utopia-payoff"))
_EXPENSIVE_METHODS: frozenset = frozenset(("shapley", "banzhaf", "s_banzhaf"))
_SKIP_METHODS: frozenset = frozenset(("nucleolus", "tau"))

# Rechenzeit-Toleranz (keine Feature-Schwelle — nur Zeitschranken)
_TIME_PROBE_MULTIPLIER: int = 500   # projected > probe_time × 500  → infeasible
_TIME_ABSOLUTE_MAX: float = 10.0    # projected > 10 s              → infeasible
_MEMORY_FRACTION: float = 0.25      # max 25 % des verfügbaren RAM nutzen


# ------------------------------------------------------------------ dataclass

@dataclass
class CoalitionPlan:
    """
    Ergebnis der Planung: welche Koalitionen vorberechnet werden,
    in welchem Modus jede Methode läuft, und warum.
    """
    coalitions: Set[Tuple]       # Koalitionen die vorberechnet werden sollen
    modes: Dict[str, str]        # method -> "exact" | "approximate" | "skip"
    skip: Set[str]               # Methoden die übersprungen werden
    feasible_exact: bool         # Ob exaktes Shapley/Banzhaf machbar ist
    reason: str                  # Lesbare Begründung für die Entscheidung
    projected_seconds: float     # Geschätzte Laufzeit für exakten Pfad


# ------------------------------------------------------ coalition helpers

def coalitions_for_method(method: str, F: np.ndarray) -> Set[Tuple]:
    """
    Gibt alle Koalitionen zurück, die eine Methode als charakteristische
    Werte benötigt.

    Gately:      v(N), v({i}), v(N\\{i})  für jedes i  →  2n+1 Koalitionen
    Utopia:      v(N), v(N\\{i})           für jedes i  →   n+1 Koalitionen
    Shapley/Ban: alle 2^n Teilmengen
    Nucl./Tau:   alle echten Teilmengen (gleich wie shapley/banzhaf)
    """
    F_int = [int(f) for f in F]
    F_tuple = tuple(F_int)
    n = len(F_int)

    if method == "gately":
        result: Set[Tuple] = {(), F_tuple}
        for i in F_int:
            result.add((i,))
            result.add(tuple(f for f in F_int if f != i))
        return result

    if method in ("utopia-payoff", "s_utopia-payoff"):
        result = {(), F_tuple}
        for i in F_int:
            result.add(tuple(f for f in F_int if f != i))
        return result

    # shapley, banzhaf, s_banzhaf, nucleolus, tau — all need the full power set
    return coalitions_for_shapley_banzhaf_exact(F)


def coalitions_for_shapley_banzhaf_exact(F: np.ndarray) -> Set[Tuple]:
    """Alle 2^n Teilmengen von F als sortierte Tupel."""
    n = len(F)
    F_int = [int(f) for f in F]
    result: Set[Tuple] = set()
    for mask in range(1 << n):
        coal = tuple(F_int[j] for j in range(n) if mask & (1 << j))
        result.add(coal)
    return result


def merge_coalition_sets(*sets: Set[Tuple]) -> Set[Tuple]:
    """Vereinigt mehrere Koalitionsmengen."""
    out: Set[Tuple] = set()
    for s in sets:
        out |= s
    return out


# --------------------------------------------------- feasibility assessment

def _read_available_memory_bytes() -> int:
    """Liest verfügbaren RAM aus /proc/meminfo. Gibt 0 zurück wenn nicht lesbar."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def assess_exact_feasibility(
    probe_fn: Callable,
    n: int,
    bytes_per_coalition: int,
    num_probe: int = 5,
) -> Tuple[bool, str, float]:
    """
    Entscheidet zur Laufzeit — ohne festen Feature-Schwellenwert — ob exakte
    Koalitionsberechnung (2^n Aufrufe) durchführbar ist.

    Zwei unabhängige Kriterien:
      A) Speicher: estimated_bytes > available_memory × 0.25  → infeasible
      B) Zeit:     projected_s    > max(probe_s × 500, 10)   → infeasible

    Args:
        probe_fn:            Callable(C: tuple) → führt eine echte v_C-Berechnung durch.
        n:                   Anzahl Features (Koalitionen = 2^n).
        bytes_per_coalition: Geschätzte Byte im Speicher pro Koalition.
        num_probe:           Anzahl Probe-Aufrufe für den Benchmark.

    Returns:
        (feasible: bool, reason: str, projected_seconds: float)
    """
    num_coalitions = 1 << n  # 2^n

    # ── A) Speicher-Check ──────────────────────────────────────────────────
    available_bytes = _read_available_memory_bytes()
    if available_bytes > 0 and bytes_per_coalition > 0:
        estimated_bytes = num_coalitions * bytes_per_coalition
        threshold_bytes = available_bytes * _MEMORY_FRACTION
        if estimated_bytes > threshold_bytes:
            reason = (
                f"Speicher: {estimated_bytes / 2**30:.2f} GB benötigt für {num_coalitions:,} "
                f"Koalitionen, aber nur {threshold_bytes / 2**30:.2f} GB erlaubt "
                f"({_MEMORY_FRACTION*100:.0f}% von {available_bytes/2**30:.2f} GB verfügbar). "
                f"→ MC-Approximation."
            )
            return False, reason, float("inf")

    # ── B) Micro-Benchmark ────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    F = np.arange(n)

    # Koalitionen verschiedener Größen (leer, viertel, halb, dreiviertel, voll)
    probe_coalitions: list = []
    for frac in np.linspace(0, 1, num_probe):
        size = int(round(frac * n))
        if size == 0:
            probe_coalitions.append(())
        elif size == n:
            probe_coalitions.append(tuple(int(f) for f in F))
        else:
            chosen = tuple(sorted(int(x) for x in rng.choice(n, size=size, replace=False)))
            probe_coalitions.append(chosen)

    t0 = time.perf_counter()
    for C in probe_coalitions:
        try:
            probe_fn(C)
        except Exception:
            pass
    probe_elapsed = time.perf_counter() - t0

    tau = probe_elapsed / max(len(probe_coalitions), 1)
    projected = tau * num_coalitions
    threshold_time = max(probe_elapsed * _TIME_PROBE_MULTIPLIER, _TIME_ABSOLUTE_MAX)

    if projected > threshold_time:
        reason = (
            f"Zeit: {projected:,.0f}s geschätzt für {num_coalitions:,} Koalitionen "
            f"({tau*1000:.3f}ms/Koalition, Schwellenwert={threshold_time:.1f}s). "
            f"→ MC-Approximation."
        )
        return False, reason, projected

    reason = (
        f"Exakter Pfad: ~{projected:.2f}s für {num_coalitions:,} Koalitionen "
        f"({tau*1000:.3f}ms/Koalition)."
    )
    return True, reason, projected


# --------------------------------------------------------- plan_computation

def plan_computation(
    methods,
    F: np.ndarray,
    probe_fn: Optional[Callable] = None,
    bytes_per_coalition: int = 0,
) -> CoalitionPlan:
    """
    Bestimmt zur Laufzeit, welche Koalitionen vorberechnet werden und in welchem
    Modus jede Methode ausgeführt wird.

    Logik:
    - Gately/Utopia  → immer sparse-exakt (O(n) Koalitionen)
    - Shapley/Banzhaf → exakt wenn machbar (Feasibility-Check), sonst MC
    - Nucleolus/Tau  → exakt wenn Shapley/Banzhaf exakt, sonst skip + warn

    Args:
        methods:              Iterable von Methodennamen (z.B. ["shapley", "gately"]).
        F:                    Feature-Array (np.arange(state_dim)).
        probe_fn:             Callable(C) für den Feasibility-Check. None → exakt annehmen.
        bytes_per_coalition:  Bytes im Speicher pro Koalition (für RAM-Check).

    Returns:
        CoalitionPlan
    """
    methods = list(methods)
    n = len(F)

    cheap = [m for m in methods if m in _SPARSE_METHODS]
    expensive = [m for m in methods if m in _EXPENSIVE_METHODS]
    skip_default = [m for m in methods if m in _SKIP_METHODS]

    # Koalitionen für sparse Methoden — immer berechnen
    cheap_coalitions: Set[Tuple] = set()
    for m in cheap:
        cheap_coalitions |= coalitions_for_method(m, F)
    cheap_coalitions.add(())   # leere Koalition immer inkludieren

    modes: Dict[str, str] = {m: "exact" for m in cheap}
    skip: Set[str] = set()
    feasible_exact = True
    reason = "Exakter Pfad (keine teuren Methoden angefragt)."
    projected_seconds = 0.0

    if expensive or skip_default:
        # Feasibility-Check nur wenn teure/skip-Methoden angefragt werden
        if probe_fn is not None:
            feasible_exact, reason, projected_seconds = assess_exact_feasibility(
                probe_fn, n, bytes_per_coalition
            )
        else:
            feasible_exact = True
            reason = "Kein probe_fn übergeben — exakter Pfad angenommen."

        if feasible_exact:
            # Alle 2^n Koalitionen — deckt Shapley/Banzhaf/Nucleolus/Tau ab
            all_coalitions = cheap_coalitions | coalitions_for_shapley_banzhaf_exact(F)
            for m in expensive:
                modes[m] = "exact"
            for m in skip_default:
                modes[m] = "exact"
        else:
            # Nur sparse Koalitionen für Gately/Utopia
            all_coalitions = cheap_coalitions
            for m in expensive:
                modes[m] = "approximate"
            for m in skip_default:
                modes[m] = "skip"
                skip.add(m)
    else:
        # Nur cheap-Methoden (Gately/Utopia) — keine teuren Koalitionen nötig
        all_coalitions = cheap_coalitions

    return CoalitionPlan(
        coalitions=all_coalitions,
        modes=modes,
        skip=skip,
        feasible_exact=feasible_exact,
        reason=reason,
        projected_seconds=projected_seconds,
    )
