import argparse

from shapley import Shapley
from utopia_payoff import UtopiaPayoff
from gately import Gately
from banzhaf import Banzhaf
from nucleolus import Nucleolus

class Explainer:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-e",
            "--explainer",
            type=str,
            default="shapley",
            choices=["shapley","utopia-payoff"],
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
        self.args = parser.parse_args()
        self.first_print = True

    def set_states(self, states_to_explain):
        if self.args.explainer == "shapley":
            self.expl = Shapley(states_to_explain)
        elif self.args.explainer == "utopia-payoff":
            self.expl = UtopiaPayoff(states_to_explain, normalized=self.args.normalize)

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