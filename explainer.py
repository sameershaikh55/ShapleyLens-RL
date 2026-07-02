import argparse
import copy
import pickle
from pathlib import Path
from typing import Literal

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

from result_analyzer import ResultAnalyzer
from result_visualizer import ResultVisualizer

from tic_tac_toe.run import tic_tac_toe_init, tic_tac_toe_get_characteristic_modes, tic_tac_toe_run
from gwa.run import gwa_init, gwa_get_characteristic_modes, gwa_run
from gwb.run import gwb_init, gwb_get_characteristic_modes, gwb_run
from gwc.run import gwc_init, gwc_get_characteristic_modes, gwc_run
from gwd.run import gwd_init, gwd_get_characteristic_modes, gwd_run
from taxi.run import taxi_init, taxi_get_characteristic_modes, taxi_run
from minesweeper.run import minesweeper_init, minesweeper_get_characteristic_modes, minesweeper_run
from frozen_lake.run import frozen_lake_init, frozen_lake_get_characteristic_modes, frozen_lake_run
from battleship.run import battleship_init, battleship_get_characteristic_modes, battleship_run

ExplainerName = Literal["shapley", "utopia-payoff", "gately", "banzhaf", "nucleolus", "tau"]
GameName = Literal["gwa", "gwb", "gwc", "gwd", "minesweeper", "taxi", "tic_tac_toe", "frozen_lake", "battleship"]

ComparisonConfig = tuple[ExplainerName, ExplainerName, GameName]

GAME_REGISTRY = {
    "tic_tac_toe": {
        "init": tic_tac_toe_init,
        "run": tic_tac_toe_run,
        "modes": tic_tac_toe_get_characteristic_modes,
    },
    "gwa": {
        "init": gwa_init,
        "run": gwa_run,
        "modes": gwa_get_characteristic_modes,
    },
    "gwb": {
        "init": gwb_init,
        "run": gwb_run,
        "modes": gwb_get_characteristic_modes,
    },
    "gwc": {
        "init": gwc_init,
        "run": gwc_run,
        "modes": gwc_get_characteristic_modes,
    },
    "gwd": {
        "init": gwd_init,
        "run": gwd_run,
        "modes": gwd_get_characteristic_modes,
    },
    "minesweeper": {
        "init": minesweeper_init,
        "run": minesweeper_run,
        "modes": minesweeper_get_characteristic_modes,
    },
    "taxi": {
        "init": taxi_init,
        "run": taxi_run,
        "modes": taxi_get_characteristic_modes,
    },
    "frozen_lake": {
        "init": frozen_lake_init,
        "run": frozen_lake_run,
        "modes": frozen_lake_get_characteristic_modes,
    },
    "battleship": {
        "init": battleship_init,
        "run": battleship_run,
        "modes": battleship_get_characteristic_modes,
        "adaptive": True,
    },
}


class Explainer:
    def __init__(
        self,
        env=None,
        agent=None,
        states_to_explain=None,
        valid_dict=None,
        instances=None,
        explainers: list[ExplainerName] | None = None,
        games: list[GameName] | None = None,
        configs: list[ComparisonConfig] | None = None,
        use_cache: bool = False,
        normalize: bool = False,
        scale_factor=1.0,
        cache_root: str = "cache",
        output_root: str = "outputs",
        **kwargs,
    ):
        if env is not None:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "-e",
                "--explainer",
                type=str,
                default="shapley",
                choices=[
                    "shapley",
                    "utopia-payoff",
                    "gately",
                    "banzhaf",
                    "nucleolus",
                    "tau",
                ],
                help="Choose one or multiple explainers.",
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
            self.states_to_explain = states_to_explain
            self.valid_dict = valid_dict
            self.instances = instances
            self.scale_factor = scale_factor
            self.normalize = normalize
            self._legacy_mode = True
            return

        if explainers is None or games is None or configs is None:
            raise ValueError(
                "Explainer requires either (env, agent, states_to_explain) "
                "or (explainers, games, configs)."
            )

        self.explainers = explainers
        self.games = games
        self.comparisons = configs
        self.use_cache = use_cache
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.cache_root = Path(cache_root)
        self.output_root = Path(output_root)
        self._legacy_mode = False

        self.results: dict[
            GameName, dict[ExplainerName, dict[str, object]]
        ] = {game: {} for game in games}

        self.characteristics_by_game: dict[GameName, dict[str, object]] = {}
        self.states_by_game: dict[GameName, object] = {}

    def _make_explainer(self, method: ExplainerName, states_to_explain):
        if method == "shapley":
            return Shapley(states_to_explain)
        if method == "utopia-payoff":
            return UtopiaPayoff(states_to_explain, normalized=self.normalize, scale_factor=1.0)
        if method == "s_utopia-payoff":
            return UtopiaPayoff(states_to_explain, normalized=self.normalize, scale_factor=self.scale_factor)
        if method == "gately":
            return Gately(states_to_explain, normalized=self.normalize)
        if method == "banzhaf":
            return Banzhaf(states_to_explain, normalized=self.normalize, scale_factor=1.0)
        if method == "s_banzhaf":
            return Banzhaf(states_to_explain, normalized=self.normalize, scale_factor=self.scale_factor)
        if method == "nucleolus":
            return Nucleolus(states_to_explain)
        if method == "tau":
            return TauValue(states_to_explain)
        raise ValueError(f"Unknown explainer: {method}")

    def set_states(self, states_to_explain):
        if self.args.explainer == "shapley":
            self.expl = Shapley(states_to_explain)
        elif self.args.explainer == "utopia-payoff":
            self.expl = UtopiaPayoff(states_to_explain, normalized=self.args.normalize)
        elif self.args.explainer == "gately":
            self.expl = Gately(states_to_explain, normalized=self.args.normalize)
        elif self.args.explainer == "banzhaf":
            self.expl = Banzhaf(states_to_explain, normalized=self.args.normalize)
        elif self.args.explainer == "nucleolus":
            self.expl = Nucleolus(states_to_explain)
        elif self.args.explainer == "tau":
            self.expl = TauValue(states_to_explain)

    def save(self, char_list, names):
        for char_data, characteristic_type in zip(char_list, names):
            if not self.args.cache:
                with open(f"{characteristic_type}.cache", "wb") as file:
                    pickle.dump(char_data, file)

    def print(self):
        analyzer = ResultAnalyzer(self.results)
        analyzer.save_pickle()
        analyzer.save_json()
        analyzer.save_csv()
        analyzer.save_summary_csv()
        analyzer.save_value_comparison_pickle(left_value="shapley", right_value="shapley")
        analyzer.save_value_comparison_csv(left_value="shapley", right_value="shapley")

        visualizer = ResultVisualizer(self.results)
        visualizer.plot_heatmaps()
        visualizer.plot_difference_heatmaps(left_value="shapley", right_value="shapley")
        visualizer.plot_value_comparison_bars()

    def get_cache(self, game_loc=None, characteristic_names=None):
        with open(game_loc + "/local.cache", "rb") as f:
            local = pickle.load(f)
        with open(game_loc + "/policy.cache", "rb") as f:
            policy = pickle.load(f)
        with open(game_loc + "/value_function.cache", "rb") as f:
            value_function = pickle.load(f)
        return [local, policy, value_function]

    def compute_state_dist(self, sample_size=1e6, agent=None, env=None):
        """Approximiert die Zustandsverteilung (wie utils.get_state_dist)."""
        ag = agent or self.agent
        ev = env or self.env
        if ag is None or ev is None:
            raise ValueError(
                "compute_state_dist: agent and environment should have been set in explainer."
            )
        self.state_dist = get_state_dist(ag, ev, int(sample_size))
        return self.state_dist

    def get_value_table_for_shapley(self, agent=None, env=None):
        ag = agent or self.agent
        ev = env or self.env
        if ag is None or ev is None:
            raise ValueError(
                "get_value_table_for_shapley: agent and environment should have been set in explainer."
            )
        ag.get_value_table(ev.valid_dict)

    def compute_pi_Cs(self, valid_dict=None, coalitions=None):
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
        """Compute partially observed value tables v_C."""
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

        Legacy mode (characteristics provided): runs each method against
        pre-computed characteristic values.

        Adaptive mode (characteristics=None): chooses exact or Monte-Carlo
        approximation at runtime via coalition planning.
        """
        if characteristics is not None:
            return self._run_values_from_characteristics(characteristics, methods, normalized)

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

        if characteristic_mode == "shapley_on_value" and not hasattr(self.agent, "value_table"):
            self.agent.get_value_table()

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

        sparse_cv: dict = {}
        for C in tqdm_label(
            sorted(cp.coalitions, key=len), "Calculating characteristic values"
        ):
            sparse_cv[C] = probe_fn(C)

        char_values_dict = {characteristic_mode: sparse_cv}

        if num_mc_samples is None:
            num_mc_samples = int(max(3500, n))
        else:
            num_mc_samples = int(num_mc_samples)

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
                    mc_cache: dict = {}

                    def char_fn_cached(C, _cache=mc_cache):
                        if C not in _cache:
                            _cache[C] = probe_fn(C)
                        return _cache[C]

                    mc_cache.update(sparse_cv)

                    if method == "shapley":
                        expl = Shapley(self.states_to_explain)
                        mc_result = expl.run_monte_carlo(char_fn_cached, num_mc_samples)
                        results[method] = {characteristic_mode: mc_result}
                    elif method in ("banzhaf", "s_banzhaf"):
                        sf = 1.0 if method == "banzhaf" else self.scale_factor
                        expl = Banzhaf(self.states_to_explain, normalized=normalized, scale_factor=sf)
                        mc_result = expl.run_monte_carlo(
                            char_fn_cached, num_mc_samples, normalized=normalized
                        )
                        results[method] = {characteristic_mode: mc_result}

                else:
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
                        elif method == "s_banzhaf":
                            expl = Banzhaf(self.states_to_explain, normalized=normalized, scale_factor=self.scale_factor)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "nucleolus":
                            expl = Nucleolus(self.states_to_explain)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "utopia-payoff":
                            expl = UtopiaPayoff(self.states_to_explain, normalized=normalized)
                            results[method][char_name] = expl.run(char_values)
                        elif method == "s_utopia-payoff":
                            expl = UtopiaPayoff(
                                self.states_to_explain,
                                normalized=normalized,
                                scale_factor=self.scale_factor,
                            )
                            results[method][char_name] = expl.run(char_values)
                        elif method == "gately":
                            expl = Gately(self.states_to_explain, normalized=normalized)
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
                elif method == "s_banzhaf":
                    expl = Banzhaf(self.states_to_explain, normalized=normalized, scale_factor=self.scale_factor)
                    results[method][char_name] = expl.run(char_values)
                elif method == "nucleolus":
                    expl = Nucleolus(self.states_to_explain)
                    results[method][char_name] = expl.run(char_values)
                elif method == "utopia-payoff":
                    expl = UtopiaPayoff(self.states_to_explain, normalized=normalized)
                    results[method][char_name] = expl.run(char_values)
                elif method == "s_utopia-payoff":
                    expl = UtopiaPayoff(
                        self.states_to_explain,
                        normalized=normalized,
                        scale_factor=self.scale_factor,
                    )
                    results[method][char_name] = expl.run(char_values)
                elif method == "gately":
                    expl = Gately(self.states_to_explain, normalized=normalized)
                    results[method][char_name] = expl.run(char_values)
                elif method == "tau":
                    expl = TauValue(self.states_to_explain)
                    results[method][char_name] = expl.run(char_values)
        return results

    def _cache_dir(self, game: GameName):
        path = self.cache_root / game
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_path(self, game: GameName, characteristic_mode):
        return self._cache_dir(game) / f"{characteristic_mode}.cache"

    def save_characteristics_cache(self, game: GameName, characteristics: dict[str, object]):
        for characteristic_mode, char_data in characteristics.items():
            path = self._cache_path(game, characteristic_mode)
            with open(path, "wb") as file:
                pickle.dump(char_data, file)

    def load_characteristics_cache(
        self, game: GameName, characteristic_modes: list[str] | None = None, strict=True
    ):
        game_path = self._cache_dir(game)

        if characteristic_modes is None:
            characteristic_modes = [p.stem for p in game_path.glob("*.cache")]

        characteristics = {}
        missing = []

        for characteristic_mode in characteristic_modes:
            path = self._cache_path(game, characteristic_mode)
            if not path.exists():
                missing.append(characteristic_mode)
                continue
            with open(path, "rb") as file:
                characteristics[characteristic_mode] = pickle.load(file)

        if strict and missing:
            raise FileNotFoundError(
                f"Missing cache files for game '{game}': {path}'/'{missing}"
            )
        return characteristics

    def _get_game_spec(self, game: GameName):
        if game not in GAME_REGISTRY:
            raise ValueError(f"No registry entry for game '{game}'")
        return GAME_REGISTRY[game]

    def _adaptive_cache_path(self, game: GameName):
        return self._cache_dir(game) / "adaptive_results.pkl"

    def _compute_or_load_adaptive_game(self, game: GameName):
        spec = self._get_game_spec(game)
        cache_path = self._adaptive_cache_path(game)

        if self.use_cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        else:
            env, agent, _ = spec["init"]()
            payload = spec["run"](
                env=env,
                agent=agent,
                states_to_explain=None,
                methods=list(self.explainers),
                normalized=self.normalize,
                scale_factor=self.scale_factor,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)

        states_to_explain = payload["states_to_explain"]
        self.states_by_game[game] = states_to_explain
        return states_to_explain, payload["results"]

    def compute_or_load_game(self, game: GameName) -> tuple[object, dict[str, object]]:
        spec = self._get_game_spec(game)

        env, agent, states_to_explain = spec["init"]()
        characteristic_modes = spec["modes"]()

        if self.use_cache:
            characteristics = self.load_characteristics_cache(
                game=game,
                characteristic_modes=characteristic_modes,
                strict=True,
            )
        else:
            characteristics = spec["run"](
                env=env,
                agent=agent,
                states_to_explain=states_to_explain,
            )
            self.save_characteristics_cache(
                game=game,
                characteristics=characteristics,
            )

        self.states_by_game[game] = states_to_explain
        self.characteristics_by_game[game] = characteristics
        return states_to_explain, characteristics

    def run_one_game(self, game: GameName):
        print(f"Running game: {game}")
        spec = self._get_game_spec(game)

        if spec.get("adaptive"):
            _, adaptive_results = self._compute_or_load_adaptive_game(game)
            char_mode = spec["modes"]()[0]
            for method in self.explainers:
                if method in adaptive_results:
                    self.results[game][method] = adaptive_results[method]
                else:
                    self.results[game][method] = {char_mode: {}}
            return

        states_to_explain, characteristics = self.compute_or_load_game(game)

        for method in self.explainers:
            expl = self._make_explainer(method, states_to_explain)

            method_results = {}
            for characteristic_type, char_data in characteristics.items():
                method_results[characteristic_type] = expl.run(char_data)

            self.results[game][method] = method_results

    def run_all_games(self):
        for game in self.games:
            self.run_one_game(game)

    def analyze_and_visualize(self):
        for game in self.games:
            game_results = self.results[game]

            if not game_results:
                continue

            game_output_dir = self.output_root / game

            analyzer = ResultAnalyzer(copy.deepcopy(game_results))

            analyzer.save_pickle(game_output_dir / "data/results.pkl")
            analyzer.save_json(game_output_dir / "data/results.json")
            analyzer.save_csv(game_output_dir / "data/results.csv")
            analyzer.save_summary_csv(game_output_dir / "data/summary.csv")

            visualizer = ResultVisualizer(copy.deepcopy(analyzer.results))

            visualizer.plot_heatmaps(
                output_dir=game_output_dir / "plots/raw"
            )

            visualizer.plot_value_comparison_bars(
                output_dir=game_output_dir / "plots/value_comparison"
            )

        for left_explainer, right_explainer, game in self.comparisons:
            if game not in self.results:
                raise ValueError(f"Game '{game}' not in results")

            game_results = self.results[game]

            if left_explainer not in game_results:
                raise ValueError(
                    f"Explainer '{left_explainer}' missing for game '{game}'"
                )
            if right_explainer not in game_results:
                raise ValueError(
                    f"Explainer '{right_explainer}' missing for game '{game}'"
                )

            pair_results = {
                left_explainer: copy.deepcopy(game_results[left_explainer]),
                right_explainer: copy.deepcopy(game_results[right_explainer]),
            }

            comparison_output_dir = (
                self.output_root
                / game
                / f"compare_{left_explainer}_vs_{right_explainer}"
            )
            print(f"comparing {left_explainer} and {right_explainer}")

            analyzer = ResultAnalyzer(pair_results)

            analyzer.save_value_comparison_pickle(
                left_value=left_explainer,
                right_value=right_explainer,
                output_dir=comparison_output_dir / "data/value_comparison.pkl",
            )

            analyzer.save_value_comparison_csv(
                left_value=left_explainer,
                right_value=right_explainer,
                output_dir=comparison_output_dir / "data/value_comparison.csv",
            )

            visualizer = ResultVisualizer(copy.deepcopy(analyzer.results))

            visualizer.plot_difference_heatmaps(
                left_value=left_explainer,
                right_value=right_explainer,
                output_dir=comparison_output_dir / "plots/diff",
            )

    def run(self):
        self.run_all_games()
        self.analyze_and_visualize()
