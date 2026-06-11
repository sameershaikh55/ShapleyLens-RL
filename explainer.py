import argparse
import pickle

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
from tic_tac_toe.run import tic_tac_toe_init, tic_tac_toe_run

class Explainer:
    def __init__(self):
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
        self.pi_Cs = {
            tuple(C): self.agent.get_pi_C(C, self.state_dist, self.states_to_explain)
            for C in tqdm_label(F_not_i(F), "Calculating all pi_C")
        }
        return self.pi_Cs

    def compute_v_Cs(self, valid_dict=None):
        """Compute partially observed value tables v_C for all coalitions C."""
        if self.agent is None or self.env is None or self.states_to_explain is None:
            self.v_Cs = {}
            return self.v_Cs
        if getattr(self, "state_dist", None) is None:
            raise ValueError("explainer.state_dist fehlt: zuerst get_state_dist(...) zuweisen.")
        if not hasattr(self.agent, "value_table"):
            raise ValueError("agent.value_table fehlt: zuerst agent.get_value_table() aufrufen.")
        F = np.arange(self.env.state_dim)
        self.v_Cs = {
            tuple(C): self.agent.get_v_C(C, self.state_dist, self.states_to_explain)
            for C in tqdm_label(F_not_i(F), "Calculating all v_C")
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
    
    def old_run(self, char_list, names):
        for char_data, characteristic_type in zip(char_list, names):
            explainer_values = self.expl.run(char_data)
            self.results[self.args.explainer][characteristic_type] = explainer_values
            with open('{}.pkl'.format(characteristic_type), 'wb') as file: pickle.dump(explainer_values, file)

    def run(self, method, characteristics):
        for char_data, characteristic_type in characteristics.items():
            explainer_values = self.expl.run(char_data)
            self.results[method][characteristic_type] = explainer_values
            with open('{}.pkl'.format(characteristic_type), 'wb') as file: pickle.dump(explainer_values, file)
    
    def run_values(self, characteristics, methods=('shapley', 'banzhaf', 'nucleolus'), normalized=True):
        """
        Run value calculations using the specified methods.
        
        Args:
            characteristics: Dictionary of characteristic functions
            methods: Tuple of method names ('shapley', 'banzhaf', 'nucleolus', etc.)
            normalized: Whether to normalize values
            
        Returns:
            Dictionary mapping method names to their results
        """
        results = {}
        
        for method in methods:
            if method == 'shapley':
                self.expl = Shapley(self.states_to_explain)
            elif method == 'banzhaf':
                self.expl = Banzhaf(self.states_to_explain, normalized=normalized)
            elif method == 'nucleolus':
                self.expl = Nucleolus(self.states_to_explain)
            elif method == 'utopia-payoff':
                self.expl = UtopiaPayoff(self.states_to_explain, normalized=normalized)
            elif method == 'gately':
                self.expl = Gately(self.states_to_explain)
            elif method == 'tau':
                self.expl = TauValue(self.states_to_explain)
            self.run(method=method, characteristics=characteristics)
        
        return results
    
if __name__ == "__main__":
    explainer = Explainer()

    env, agent, states_to_explain = tic_tac_toe_init()
    explainer.env = env
    explainer.agent = agent
    explainer.states_to_explain = states_to_explain

    if explainer.args.cache:
        characteristic_names = ["local", "policy", "value_function"]
        try:
            characteristic_results = explainer.get_cache(game_loc="tic_tac_toe")
        except:
            print("Cache files not found. Please run without --cached True first")
            sys.exit(1)
    else:
        characteristic_results, characteristic_names = tic_tac_toe_run(env=env, agent=agent, states_to_explain=states_to_explain)
        # characteristics = 

    explainer.set_states(states_to_explain)

    explainer.old_run(char_list=characteristic_results, names=characteristic_names)
    explainer.print()