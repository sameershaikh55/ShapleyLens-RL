import numpy as np
from scipy.optimize import linprog
from collections import defaultdict
from utils import F_not_i

# Tolerance for identifying tight (saturated) coalitions at each LP step.
_TOL = 1e-7


class Nucleolus:
    """
    Calculates Nucleolus values given characteristic values.

    Mirrors the interface of the Shapley class for plug-and-play use
    in the SVERL framework:
        nucleolus = Nucleolus(states_to_explain)
        nuc_values = nucleolus.run(characteristic_values)

    ---------------------------------------------------------------------------
    Mathematical Background
    ---------------------------------------------------------------------------
    For a cooperative game (N, v) with N = {0,...,n-1} features and
    characteristic function v: 2^N -> R, the *excess* of coalition S under
    payoff vector x is:
        e(S, x) = v(S) - sum_{i in S} x_i

    The Nucleolus x* is the unique imputation that lexicographically minimises
    the vector of all coalition excesses sorted in non-increasing order.

    It is computed via Maschler's sequential LP algorithm:

    Step 0  Active set A = all proper non-empty subsets of N.
            Equality constraints E = {sum(x) = v(N)}  (efficiency).

    Step k  Solve LP_k:
               min  eps
               s.t. v(S) - x(S) <= eps      for all S in A  (ineq.)
                    [equality constraints in E]
            Let eps* be the optimal value and x* the optimal allocation.

    Step k+1 Identify the 'saturated' (tight) coalitions:
                 B = {S in A : v(S) - x*(S) >= eps* - tol}
             These coalitions must achieve exactly eps* in every nucleolus
             solution, so pin them:
                 E  <-  E  +  {x(S) = v(S) - eps*  for each S in B}
                 A  <-  A minus B
             Repeat from Step k until A is empty.

    The final x* is the Nucleolus.

    Properties guaranteed by this algorithm:
        * Efficiency:     sum(x_i) = v(N)
        * Unique:         the Nucleolus is always unique
    """

    def __init__(self, states_to_explain):
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain

        # All proper non-empty subsets  {}  <  S  <  N
        all_C = F_not_i(self.F)          # includes [] and the full set
        self._proper_subsets = [
            tuple(C) for C in all_C
            if 0 < len(C) < self.F_card
        ]

        # Binary indicator vectors  ind[S][i] = 1 iff i in S
        self._indicators = {
            S: np.array([1.0 if i in S else 0.0 for i in range(self.F_card)])
            for S in self._proper_subsets
        }

    # ---------------------------------------------------------------------- #
    #  Public interface                                                        #
    # ---------------------------------------------------------------------- #

    def run(self, characteristic_values):
        """
        Compute the Nucleolus for every state in states_to_explain.

        Parameters
        ----------
        characteristic_values : dict
            {tuple(C): {tuple(state): float}}
            Identical format to what Characteristics produces and Shapley
            consumes.  The empty coalition () and grand coalition tuple(F)
            must both be present.

        Returns
        -------
        dict  {tuple(state): np.ndarray of shape (F_card,)}
            Nucleolus payoff vector for each explained state.
        """
        grand_C = tuple(self.F)
        empty_C = ()           # empty coalition key
        nucleolus_values = {}

        for state in self.states:
            # Flatten to {coalition_tuple: value} for this one state.
            # 'value' may be a scalar (local/global/value_function) or a
            # numpy array (shapley_on_policy, where each coalition stores a
            # probability distribution over actions).
            C_values = {
                C: vt[tuple(state)]
                for C, vt in characteristic_values.items()
            }
            v_N = C_values.get(grand_C, 0.0)

            if isinstance(v_N, np.ndarray) and v_N.ndim >= 1:
                # Vector-valued game (e.g. policy characteristic).
                # The Nucleolus is computed independently for each component;
                # this matches how Shapley handles the same characteristic
                # via element-wise arithmetic.
                n_components = v_N.size
                per_comp = []
                for k in range(n_components):
                    scalar_C_values = {
                        C: (float(val.flat[k]) if isinstance(val, np.ndarray) else float(val))
                        for C, val in C_values.items()
                    }
                    # ---------- 0-normalisation (per component) ----------
                    # Shapley's formula uses v(∅) as its baseline, so
                    # ∑ φ_i = v(N) − v(∅).  To make the Nucleolus use the
                    # same baseline we shift every coalition value by v(∅)
                    # before the LP, so ṽ(S) = v(S) − v(∅) and ṽ(∅) = 0.
                    # The LP's efficiency constraint then becomes
                    # ∑ x_i = ṽ(N) = v(N) − v(∅),  matching Shapley.
                    # For n = 2 features this makes the two solutions
                    # mathematically identical (a well-known GT result).
                    v_empty_k = scalar_C_values.get(empty_C, 0.0)
                    norm_C_values = {C: v - v_empty_k for C, v in scalar_C_values.items()}
                    v_N_norm_k    = float(v_N.flat[k]) - v_empty_k
                    per_comp.append(self._compute(norm_C_values, v_N_norm_k))
                # Shape (n_components, F_card) mirrors Shapley's policy layout.
                nucleolus_values[tuple(state)] = np.array(per_comp)
            else:
                # ---------- 0-normalisation (scalar game) ----------
                # Same logic as above: subtract v(∅) from every coalition
                # so the Nucleolus efficiency axiom matches Shapley's.
                v_empty   = float(C_values.get(empty_C, 0.0))
                norm_C_values = {C: float(v) - v_empty for C, v in C_values.items()}
                v_N_norm  = float(v_N) - v_empty
                nucleolus_values[tuple(state)] = self._compute(norm_C_values, v_N_norm)

        return nucleolus_values

    # ---------------------------------------------------------------------- #
    #  Verification helper                                                     #
    # ---------------------------------------------------------------------- #

    def verify_efficiency(self, nucleolus_values, characteristic_values):
        """
        Print and return whether sum(x_i) == v(N) for every explained state.

        This is the primary correctness check: the Nucleolus must always
        satisfy efficiency.  A passing check does NOT prove the lexicographic
        minimality, but a failing check is an immediate red flag.

        Parameters
        ----------
        nucleolus_values     : output of run()
        characteristic_values: same dict passed to run()

        Returns
        -------
        bool  True if all states pass within 1e-5 tolerance.
        """
        grand_C = tuple(self.F)
        print("\n[Nucleolus] Efficiency verification  (sum x_i == v(N)):")
        all_ok = True

        for state in self.states:
            nuc  = nucleolus_values[tuple(state)]
            v_N  = characteristic_values[grand_C][tuple(state)]

            # Vector-valued characteristic: check efficiency per component.
            if isinstance(nuc, np.ndarray) and nuc.ndim == 2:
                # nuc shape: (n_components, F_card); each row must sum to v_N[k]
                for k in range(nuc.shape[0]):
                    comp_sum = float(np.sum(nuc[k]))
                    comp_vN  = float(v_N.flat[k]) if isinstance(v_N, np.ndarray) else float(v_N)
                    diff = abs(comp_sum - comp_vN)
                    ok   = diff < 1e-5
                    flag = "OK" if ok else "FAIL"
                    print(
                        f"  State {str(tuple(state)):20s}  component[{k}]  "
                        f"sum={comp_sum:+.6f}  v(N)={comp_vN:+.6f}  "
                        f"|error|={diff:.2e}  [{flag}]"
                    )
                    if not ok:
                        all_ok = False
            else:
                diff = abs(float(np.sum(nuc)) - float(v_N))
                ok   = diff < 1e-5
                flag = "OK" if ok else "FAIL"
                print(
                    f"  State {str(tuple(state)):20s}  "
                    f"sum={np.sum(nuc):+.6f}  "
                    f"v(N)={float(v_N):+.6f}  "
                    f"|error|={diff:.2e}  [{flag}]"
                )
                if not ok:
                    all_ok = False

        return all_ok

    # ---------------------------------------------------------------------- #
    #  Core LP solver                                                          #
    # ---------------------------------------------------------------------- #

    def _compute(self, C_values, v_N):
        """
        Maschler's iterative LP to compute the Nucleolus.

        Decision variables  y = [x_0, ..., x_{n-1}, eps]  (length n+1)
        """
        n = self.F_card

        # Handle trivial single-feature case
        if n == 1 or not self._proper_subsets:
            return np.array([v_N])

        # Objective: minimise eps  (last variable)
        c      = np.zeros(n + 1)
        c[-1]  = 1.0

        # All variables unbounded
        bounds = [(None, None)] * (n + 1)

        # ------------------------------------------------------------------ #
        # Equality constraint list (grows as coalitions get saturated)        #
        # ------------------------------------------------------------------ #
        # Efficiency:  x_0 + ... + x_{n-1}  = v(N)
        eff_row       = np.zeros(n + 1)
        eff_row[:n]   = 1.0
        A_eq_list     = [eff_row]
        b_eq_list     = [v_N]

        active  = list(self._proper_subsets)   # coalitions not yet pinned
        x_sol   = np.full(n, v_N / n)          # fallback allocation

        while active:
            # -------------------------------------------------------------- #
            # Build inequality block: v(S) - x(S) <= eps                     #
            #   =>  -ind @ x  -  eps  <=  -v(S)                              #
            # -------------------------------------------------------------- #
            A_ub_list = []
            b_ub_list = []
            for S in active:
                row       = np.zeros(n + 1)
                row[:n]   = -self._indicators[S]
                row[-1]   = -1.0
                A_ub_list.append(row)
                b_ub_list.append(-C_values.get(S, 0.0))

            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
            A_eq = np.array(A_eq_list)
            b_eq = np.array(b_eq_list)

            result = linprog(
                c,
                A_ub=A_ub,  b_ub=b_ub,
                A_eq=A_eq,  b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'disp': False, 'presolve': True},
            )

            if not result.success:
                # LP infeasible / unbounded – keep best allocation found so far
                print(
                    f"  [Nucleolus] Warning: LP failed at iteration "
                    f"({len(active)} active coalitions remain). "
                    f"Status: {result.message}"
                )
                break

            eps_star = float(result.x[-1])
            x_sol    = result.x[:n].copy()

            # -------------------------------------------------------------- #
            # Identify saturated coalitions                                   #
            # Use a scale-adaptive tolerance to handle large/small rewards    #
            # -------------------------------------------------------------- #
            tol = max(_TOL, 1e-6 * abs(eps_star))

            tight     = []
            remaining = []
            for S in active:
                excess = C_values.get(S, 0.0) - float(
                    np.dot(self._indicators[S], x_sol)
                )
                # At optimum all excesses <= eps_star; tight means excess ≈ eps_star
                if excess >= eps_star - tol:
                    tight.append(S)
                else:
                    remaining.append(S)

            # Safety: if nothing identified as tight (pure numerical drift),
            # force the single highest-excess coalition to be tight so the
            # algorithm always makes progress.
            if not tight:
                best = max(
                    active,
                    key=lambda S: C_values.get(S, 0.0)
                    - float(np.dot(self._indicators[S], x_sol)),
                )
                tight     = [best]
                remaining = [S for S in active if S != best]

            # -------------------------------------------------------------- #
            # Pin tight coalitions:  x(S) = v(S) - eps*                      #
            # They become hard equality constraints for all future LPs.       #
            # -------------------------------------------------------------- #
            for S in tight:
                row      = np.zeros(n + 1)
                row[:n]  = self._indicators[S]
                A_eq_list.append(row)
                b_eq_list.append(C_values.get(S, 0.0) - eps_star)

            active = remaining

        return x_sol
