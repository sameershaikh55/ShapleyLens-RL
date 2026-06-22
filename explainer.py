import os
import copy
import argparse
import pickle
from pathlib import Path
from typing import Literal
from contextlib import contextmanager

from shapley import Shapley
from utopia_payoff import UtopiaPayoff
from gately import Gately
from banzhaf import Banzhaf
from nucleolus import Nucleolus
from tau import TauValue

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
from tic_tac_toe.run import tic_tac_toe_init, tic_tac_toe_get_characteristic_modes, tic_tac_toe_run
from gwa.run import gwa_init, gwa_get_characteristic_modes, gwa_run
from gwb.run import gwb_init, gwb_get_characteristic_modes, gwb_run
from gwc.run import gwc_init, gwc_get_characteristic_modes, gwc_run
from gwd.run import gwd_init, gwd_get_characteristic_modes, gwd_run
from taxi.run import taxi_init, taxi_get_characteristic_modes, taxi_run
from minesweeper.run import minesweeper_init, minesweeper_get_characteristic_modes, minesweeper_run
from frozen_lake.run import frozen_lake_init, frozen_lake_get_characteristic_modes, frozen_lake_run

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
        "modes": gwa_get_characteristic_modes
    },
    "gwb": {
        "init": gwb_init,
        "run": gwb_run,
        "modes": gwb_get_characteristic_modes
    },
    "gwc": {
        "init": gwc_init,
        "run": gwc_run,
        "modes": gwc_get_characteristic_modes
    },
    "gwd": {
        "init": gwd_init,
        "run": gwd_run,
        "modes": gwd_get_characteristic_modes
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
}

class Explainer:

    def __init__(
        self,
        explainers: list[ExplainerName],
        games: list[GameName],
        configs: list[ComparisonConfig],
        use_cache: bool = False,
        normalize: bool = False,
        cache_root: str = "cache",
        output_root: str = "outputs",
    ):

        self.explainers = explainers
        self.games = games
        self.comparisons = configs
        self.use_cache = use_cache
        self.normalize = normalize
        self.cache_root = Path(cache_root)
        self.output_root = Path(output_root)

        self.results: dict[
            GameName, dict[ExplainerName, dict[str, object]]
        ] = {game: {} for game in games}

        self.characteristics_by_game: dict[GameName, dict[str, object]] = {}
        self.states_by_game: dict[GameName, object] = {}

    def _make_explainer(
        self, method: ExplainerName, states_to_explain
    ):
        if method == "shapley":
            return Shapley(states_to_explain)
        if method == "utopia-payoff":
            return UtopiaPayoff(states_to_explain, normalized=self.normalize)
        if method == "gately":
            return Gately(states_to_explain, normalized=self.normalize)
        if method == "banzhaf":
            return Banzhaf(states_to_explain, normalized=self.normalize)
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

    def load_characteristics_cache(self, game: GameName, characteristic_modes : list[str] | None = None, strict=True):
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
    
    def compute_or_load_game(
        self,
        game: GameName,
    ) -> tuple[object, dict[str, object]]:
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