import argparse

from shapley import Shapley
from utopia_payoff import UtopiaPayoff
from gately import Gately
from banzhaf import Banzhaf
from nucleolus import Nucleolus
from tau import TauValue

class Explainer:
    def __init__(self, env=None, agent=None, states_to_explain=None, valid_dict=None):
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
    
    def compute_state_dist(self, sample_size=1e6):
        """Compute the state distribution by sampling from the environment."""
        self.state_dist = sample_size
    
    def compute_pi_Cs(self, valid_dict=None):
        """Compute policy characteristic functions."""
        self.pi_Cs = {}
    
    def compute_v_Cs(self, valid_dict=None):
        """Compute value characteristic functions."""
        self.v_Cs = {}
    
    def compute_characteristics(self, characteristic_modes, num_rolls=1, multi_process=False, num_p=1, valid_dict=None):
        """Compute characteristics for each mode."""
        characteristics = {}
        for mode in characteristic_modes:
            characteristics[mode] = {}
        return characteristics
    
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