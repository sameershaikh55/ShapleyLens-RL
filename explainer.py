import argparse

import numpy as np

from shapley import Shapley
from utopia_payoff import UtopiaPayoff
from gately import Gately
from banzhaf import Banzhaf
from nucleolus import Nucleolus
from tau import TauValue
from characteristics import Characteristics
from utils import F_not_i, tqdm_label


class Explainer:
    def __init__(self, env=None, agent=None, states_to_explain=None, valid_dict=None, instances=None, **kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-e",
            "--explainer",
            type=str,
            default="shapley",
            choices=["shapley","utopia-payoff", "gately", "banzhaf", "nucleolus", "tau"],
            help="Choose explainer",
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
        # Use parse_known_args to avoid conflicts with other CLI arguments
        self.args, _ = parser.parse_known_args()
        self.first_print = True
        
        # Store environment and agent info
        self.env = env
        self.agent = agent
        self.states_to_explain = states_to_explain
        self.valid_dict = valid_dict
        self.instances = instances

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

    def run(self, characteristics):            
        return self.expl.run(characteristics)
    
    def print(self, explainer_values, characteristic_type):
        if self.first_print == True:
            self.first_print = False
            print(self.args.explainer.replace("_", " ").title() + ":")
        print(explainer_values)

    def get_cache(self):
        import pickle

        with open("local.cache", "rb") as f:
            local = pickle.load(f)
        with open("policy.cache", "rb") as f:
            policy = pickle.load(f)
        with open("value_function.cache", "rb") as f:
            value_function = pickle.load(f)

        return local, policy, value_function
    
    def compute_state_dist(self, sample_size=1e6, agent=None, env=None):
        """Approximiert die Zustandsverteilung (wie utils.get_state_dist)."""
        from utils import get_state_dist

        ag = agent or self.agent
        ev = env or self.env
        if ag is None or ev is None:
            raise ValueError("compute_state_dist: agent und env müssen am Explainer gesetzt oder übergeben sein.")
        self.state_dist = get_state_dist(ag, ev, int(sample_size))
        return self.state_dist

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
                explainer = Shapley(self.states_to_explain)
                results[method] = {}
                for char_name, char_values in characteristics.items():
                    results[method][char_name] = explainer.run(char_values) if char_values else {}
            elif method == 'banzhaf':
                explainer = Banzhaf(self.states_to_explain, normalized=normalized)
                results[method] = {}
                for char_name, char_values in characteristics.items():
                    results[method][char_name] = explainer.run(char_values) if char_values else {}
            elif method == 'nucleolus':
                explainer = Nucleolus(self.states_to_explain)
                results[method] = {}
                for char_name, char_values in characteristics.items():
                    results[method][char_name] = explainer.run(char_values) if char_values else {}
            elif method == 'utopia-payoff':
                explainer = UtopiaPayoff(self.states_to_explain, normalized=normalized)
                results[method] = {}
                for char_name, char_values in characteristics.items():
                    results[method][char_name] = explainer.run(char_values) if char_values else {}
            elif method == 'gately':
                explainer = Gately(self.states_to_explain)
                results[method] = {}
                for char_name, char_values in characteristics.items():
                    results[method][char_name] = explainer.run(char_values) if char_values else {}
            elif method == 'tau':
                explainer = TauValue(self.states_to_explain)
                results[method] = {}
                for char_name, char_values in characteristics.items():
                    results[method][char_name] = explainer.run(char_values) if char_values else {}
        
        return results