import argparse

from shapley import Shapley
from utopia_payoff import UtopiaPayoff

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
        self.args = parser.parse_args()
        self.first_print = True

    def set_states(self, states_to_explain):
        if self.args.explainer == "shapley":
            self.expl = Shapley(states_to_explain)
        elif self.args.explainer == "utopia-payoff":
            self.expl = UtopiaPayoff(states_to_explain)

    def run(self, characteristics):            
        return self.expl.run(characteristics)
    
    def print(self, explainer_values, characteristic_type):
        if self.first_print == True:
            self.first_print = False
            print(self.args.explainer.replace("_", " ").title() + ":")
        print(explainer_values)        