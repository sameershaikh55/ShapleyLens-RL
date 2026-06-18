import argparse
import pickle

import numpy as np

from banzhaf import Banzhaf
from characteristics import Characteristics
from coalition_plan import CoalitionPlan, plan_computation
from gately import Gately
from nucleolus import Nucleolus
from shapley import Shapley
from tau import TauValue
from utopia_payoff import UtopiaPayoff
from utils import F_not_i, tqdm_label


import sys
sys.path.insert(0, '../')
from q_agent_2 import Agent
from tic_tac_toe.tic_tac_toe import TTT
from taxi.taxi_wrap import FactoredState
from utils import train, get_state_dist, F_not_i, tqdm_label, value_iteration, find_states_taxi
from characteristics import Characteristics
import numpy as np

from result_analyzer import ResultAnalyzer
from result_visualizer import ResultVisualizer

from tic_tac_toe.tic_tac_toe import TTT
from tic_tac_toe.run import tic_tac_toe_init, tic_tac_toe_run

class Explainer:
    def __init__(self, env=None, agent=None, states_to_explain=None, valid_dict=None, instances=None, **kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-e",
            "--explainer",
            type=str,
            # nargs="+",
            default="shapley",
            choices=[
                "shapley",
                "utopia-payoff",
                "gately",
                "banzhaf",
                "nucleolus",
                "tau"
            ],
            help="Choose one or multiple explainers. Use '-e shapley' or '-e shapley utopia-payoff tau'",
        )
        parser.add_argument(
            "-c",
            "--cache",
            action="store_true",
            help="Use cache for characteristics",
        )
        parser.add_argument(
            "-n",
            "--normalize",
            action="store_true",
            help="Normalize value",
        )

        self.args = parser.parse_args()
        self.results = {}
        self.results[self.args.explainer] = {}
        self.agent = agent
        self.env = env
        self.states_to_explain=states_to_explain
        # for i in self.args.explainers:
        #     self.results[self.args.explainers[i]] = {}


    def set_states(self, states_to_explain):
        if self.args.explainer == "shapley":
            self.expl = Shapley(states_to_explain)
        elif self.args.explainer == "utopia-payoff":
            self.expl = UtopiaPayoff(states_to_explain, normalized=self.args.normalize)
        elif self.args.explainer == "gately":
            self.expl = Gately(states_to_explain)
        elif self.args.explainer == "banzhaf":
            self.expl = Banzhaf(states_to_explain, normalized=self.args.normalize)
        elif self.args.explainer == "nucleolus":
            self.expl = Nucleolus(states_to_explain)
        elif self.args.explainer == "tau":
            self.expl = TauValue(states_to_explain)
    
    def save(self, char_list, names):
        for char_data, characteristic_type in zip(char_list, names):
            if not self.args.cache:
                with open('{}.cache'.format(characteristic_type), 'wb') as file: pickle.dump(char_data, file)
    
    def print(self):
        # ------------------------------------------------- ANALYZE RESULTS
        analyzer = ResultAnalyzer(self.results)

        analyzer.save_pickle()
        analyzer.save_json()
        analyzer.save_csv()
        analyzer.save_summary_csv()

        analyzer.save_value_comparison_pickle(left_value="shapley", right_value="shapley")
        analyzer.save_value_comparison_csv(left_value="shapley", right_value="shapley")

        # ------------------------------------------------- VISUALIZE RESULTS
        visualizer = ResultVisualizer(self.results)

        visualizer.plot_heatmaps()
        visualizer.plot_difference_heatmaps(left_value="shapley", right_value="shapley")
        visualizer.plot_value_comparison_bars()

    def get_cache(self, game_loc=None, characteristic_names=None):
        import pickle

        with open(game_loc + "/local.cache", "rb") as f:
            local = pickle.load(f)
        with open(game_loc + "/policy.cache", "rb") as f:
            policy = pickle.load(f)
        with open(game_loc + "/value_function.cache", "rb") as f:
            value_function = pickle.load(f)

        return [local, policy, value_function]
    
    def compute_state_dist(self, sample_size=1e6, agent=None, env=None):
        """Approximiert die Zustandsverteilung (wie utils.get_state_dist)."""
        from utils import get_state_dist

        ag = agent or self.agent
        ev = env or self.env
        if ag is None or ev is None:
            raise ValueError("compute_state_dist: agent and environment should have been set in explainer.")
        self.state_dist = get_state_dist(ag, ev, int(sample_size))
        return self.state_dist
    
    def get_value_table_for_shapley(self, agent=None, env=None):
        ag = agent or self.agent
        ev = env or self.env
        if ag is None or ev is None:
            raise ValueError("get_value_table_for_shapley: agent and environment should have been set in explainer.")
        ag.get_value_table(ev.valid_dict)

    def compute_pi_Cs(self, valid_dict=None):
        """Compute policy characteristic functions pi_C for all coalitions C."""
        if self.agent is None or self.env is None or self.states_to_explain is None:
            self.pi_Cs = {}
            return self.pi_Cs
        if getattr(self, "state_dist", None) is None:
            raise ValueError("explainer.state_dist fehlt: zuerst get_state_dist(...) zuweisen.")
        F = np.arange(self.env.state_dim)
        coal_iter = (
            sorted(coalitions, key=len) if coalitions is not None else F_not_i(F)
        )
        self.pi_Cs = {
            tuple(C): self.agent.get_pi_C(list(C), self.state_dist, self.states_to_explain)
            for C in tqdm_label(coal_iter, "Calculating pi_C")
        }
        return self.pi_Cs

    def compute_v_Cs(self, valid_dict=None, coalitions=None):
        """Compute partially observed value tables v_C.

        Args:
            coalitions: Optional set/iterable of coalition tuples to compute.
                        None → all 2^n coalitions (original behaviour).
        """

        if self.agent is None or self.env is None or self.states_to_explain is None:
            self.v_Cs = {}
            return self.v_Cs
        if getattr(self, "state_dist", None) is None:
            raise ValueError("explainer.state_dist fehlt: zuerst get_state_dist(...) zuweisen.")
        if not hasattr(self.agent, "value_table"):
            raise ValueError("agent.value_table fehlt: zuerst agent.get_value_table() aufrufen.")
        F = np.arange(self.env.state_dim)
        coal_iter = (
            sorted(coalitions, key=len) if coalitions is not None else F_not_i(F)
        )
        self.v_Cs = {
            tuple(C): self.agent.get_v_C(list(C), self.state_dist, self.states_to_explain)
            for C in tqdm_label(coal_iter, "Calculating v_C")
        }
        return self.v_Cs

    def compute_characteristics(
        self, characteristic_modes, num_rolls=1, multi_process=False, num_p=1, valid_dict=None
    ):
        """Compute characteristic values for each requested mode."""
        if self.env is None or self.states_to_explain is None:
            return {mode: {} for mode in characteristic_modes}
        vd = valid_dict if valid_dict is not None else self.valid_dict
        ch = Characteristics(self.env, self.states_to_explain, instances=self.instances)
        num_rolls = int(num_rolls)
        out = {mode: {} for mode in characteristic_modes}

        if "local_sverl" in characteristic_modes:
            out["local_sverl"] = ch.local_sverl_C_values(
                num_rolls, self.pi_Cs, multi_process=multi_process, num_p=num_p
            )
        if "global_sverl" in characteristic_modes:
            out["global_sverl"] = ch.global_sverl_C_values(
                num_rolls, self.pi_Cs, multi_process=multi_process, num_p=num_p
            )
        if "fast_local_sverl" in characteristic_modes:
            out["fast_local_sverl"] = ch.fast_local_sverl_C_values(
                self.pi_Cs,
                num_rolls=num_rolls,
                valid_dict=vd,
                multi_process=multi_process,
                num_p=num_p,
            )
        if "shapley_on_policy" in characteristic_modes:
            out["shapley_on_policy"] = ch.shapley_on_policy(
                self.pi_Cs, multi_process=multi_process, num_p=num_p
            )
        if "shapley_on_value" in characteristic_modes:
            out["shapley_on_value"] = ch.shapley_on_value(
                self.v_Cs, multi_process=multi_process, num_p=num_p
            )

        return out
    
    def run_values(
        self,
        characteristics=None,
        methods=("shapley", "banzhaf", "nucleolus"),
        normalized=True,
        characteristic_mode="shapley_on_value",
        num_mc_samples=None,
    ):
        """
        Run value calculations using the specified methods.

        Two operating modes:

        **Legacy mode** (characteristics provided):
            Pass a pre-computed characteristics dict → each method runs exactly
            against those characteristic values.  Fully backward-compatible.

        **Adaptive mode** (characteristics=None):
            Determines at runtime whether to run exact or Monte-Carlo
            approximation for Shapley/Banzhaf, and only pre-computes the sparse
            coalition set needed for Gately/Utopia.
            Nucleolus/Tau are skipped (with a warning entry) when the exact
            path is not feasible.

        Args:
            characteristics:    Pre-computed characteristic values dict, or None
                                to trigger the adaptive pipeline.
            methods:            Tuple/list of method names.
            normalized:         Whether to normalise Banzhaf values.
            characteristic_mode: Which characteristic to use for the adaptive
                                path: "shapley_on_value" (default) or
                                "shapley_on_policy".
            num_mc_samples:     MC samples for Shapley/Banzhaf approximation.
                                None → max(100, n) (auto).

        Returns:
            dict  method → {char_name → {state_tuple → values}}
            Plus a special "_meta" key with plan information (adaptive mode only).
        """
        if characteristics is not None:
            return self._run_values_from_characteristics(characteristics, methods, normalized)

        # ── Adaptive path ─────────────────────────────────────────────────
        if self.agent is None or self.env is None or self.states_to_explain is None:
            raise ValueError(
                "run_values (adaptive): agent, env und states_to_explain müssen gesetzt sein."
            )
        if getattr(self, "state_dist", None) is None:
            raise ValueError(
                "run_values (adaptive): state_dist fehlt — zuerst compute_state_dist() aufrufen."
            )

        n = self.env.state_dim
        F = np.arange(n)

        # Ensure value_table exists for shapley_on_value
        if characteristic_mode == "shapley_on_value" and not hasattr(self.agent, "value_table"):
            self.agent.get_value_table()

        # probe_fn for feasibility check and on-demand MC evaluation
        if characteristic_mode == "shapley_on_value":
            def probe_fn(C):
                return self.agent.get_v_C(
                    list(C), self.state_dist, self.states_to_explain
                )
            bytes_per_coalition = len(self.states_to_explain) * 8
        else:
            def probe_fn(C):
                return self.agent.get_pi_C(
                    list(C), self.state_dist, self.states_to_explain
                )
            bytes_per_coalition = len(self.state_dist) * self.env.num_actions * 8

        cp: CoalitionPlan = plan_computation(
            methods, F, probe_fn, bytes_per_coalition
        )

        print(f"[Explainer] Koalitionsplan: {cp.reason}")
        for m, mode in cp.modes.items():
            print(f"  {m}: {mode}")

        # Pre-compute the (sparse or full) characteristic values dict
        sparse_cv: dict = {}
        for C in tqdm_label(
            sorted(cp.coalitions, key=len), "Calculating characteristic values"
        ):
            sparse_cv[C] = probe_fn(C)

        char_values_dict = {characteristic_mode: sparse_cv}

        # MC sample count (auto)
        if num_mc_samples is None:
            num_mc_samples = max(35e3, n)

        results: dict = {}
        method_errors: dict = {}

        for method in methods:
            mode = cp.modes.get(method, "skip")

            if mode == "skip":
                method_errors[method] = (
                    f"Übersprungen: exakte Berechnung bei n={n} Features nicht machbar "
                    f"({cp.reason})"
                )
                continue

            try:
                if mode == "approximate":
                    # MC path — on-demand evaluation with the cached probe_fn
                    mc_cache: dict = {}

                    def char_fn_cached(C, _cache=mc_cache):
                        if C not in _cache:
                            _cache[C] = probe_fn(C)
                        return _cache[C]

                    # Seed the cache with already-computed sparse values
                    mc_cache.update(sparse_cv)

                    if method == "shapley":
                        expl = Shapley(self.states_to_explain)
                        mc_result = expl.run_monte_carlo(char_fn_cached, num_mc_samples)
                        results[method] = {characteristic_mode: mc_result}
                    elif method == "banzhaf":
                        expl = Banzhaf(self.states_to_explain, normalized=normalized)
                        mc_result = expl.run_monte_carlo(
                            char_fn_cached, num_mc_samples, normalized=normalized
                        )
                        results[method] = {characteristic_mode: mc_result}

                else:
                    # Exact path
                    results[method] = {}
                    for char_name, char_values in char_values_dict.items():
                        if not char_values:
                            results[method][char_name] = {}
                            continue
                        if method == "shapley":
                            expl = Shapley(self.states_to_explain)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "banzhaf":
                            expl = Banzhaf(self.states_to_explain, normalized=normalized)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "nucleolus":
                            expl = Nucleolus(self.states_to_explain)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "utopia-payoff":
                            expl = UtopiaPayoff(self.states_to_explain, normalized=normalized)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "gately":
                            expl = Gately(self.states_to_explain)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "tau":
                            expl = TauValue(self.states_to_explain)
                            results[method][char_name] = expl.run(char_values)

            except Exception as exc:
                method_errors[method] = str(exc)

        results["_meta"] = {
            "plan": cp,
            "method_errors": method_errors,
            "characteristic_mode": characteristic_mode,
            "num_mc_samples": num_mc_samples if not cp.feasible_exact else None,
        }
        return results

    def _run_values_from_characteristics(self, characteristics, methods, normalized):
        """Legacy exact path: runs each method against pre-computed characteristics."""
        results = {}
        for method in methods:
            results[method] = {}
            for char_name, char_values in characteristics.items():
                if not char_values:
                    results[method][char_name] = {}
                    continue
                if method == "shapley":
                    expl = Shapley(self.states_to_explain)
                    results[method][char_name] = expl.run(char_values)
                elif method == "banzhaf":
                    expl = Banzhaf(self.states_to_explain, normalized=normalized)
                    results[method][char_name] = expl.run(char_values)
                elif method == "nucleolus":
                    expl = Nucleolus(self.states_to_explain)
                    results[method][char_name] = expl.run(char_values)
                elif method == "utopia-payoff":
                    expl = UtopiaPayoff(self.states_to_explain, normalized=normalized)
                    results[method][char_name] = expl.run(char_values)
                elif method == "gately":
                    expl = Gately(self.states_to_explain)
                    results[method][char_name] = expl.run(char_values)
                elif method == "tau":
                    expl = TauValue(self.states_to_explain)
                    results[method][char_name] = expl.run(char_values)
        return results

