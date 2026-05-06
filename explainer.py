import inspect
import numpy as np
from utils import get_state_dist, F_not_i
from characteristics import Characteristics
from shapley import Shapley
from banzhaf import Banzhaf
from nucleolus import Nucleolus


class Explainer:
    """
    General explainer that calculates characteristic values and then computes
    Shapley and Banzhaf values for any compatible game.
    """

    def __init__(self, env, agent, states_to_explain, instances=None, valid_dict=None):
        self.env = env
        self.agent = agent
        self.states_to_explain = np.array(states_to_explain)
        self.instances = instances
        self.valid_dict = valid_dict if valid_dict is not None else getattr(env, 'valid_dict', None)
        self.state_dim = env.state_dim
        self.F = np.arange(self.state_dim)
        self.characteristics = Characteristics(env, self.states_to_explain, instances=instances)
        self.state_dist = None
        self.pi_Cs = None
        self.v_Cs = None

    def _agent_call(self, method_name, *args, **kwargs):
        method = getattr(self.agent, method_name)
        signature = inspect.signature(method)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return method(*args, **filtered_kwargs)

    def compute_state_dist(self, sample_size=100000):
        self.state_dist = get_state_dist(self.agent, self.env, sample_size)
        return self.state_dist

    def compute_pi_Cs(self, valid_dict=None):
        if self.state_dist is None:
            raise ValueError('State distribution must be calculated before computing pi_Cs.')

        valid_dict = valid_dict if valid_dict is not None else self.valid_dict
        self.pi_Cs = {
            tuple(C): self._agent_call('get_pi_C', C, self.state_dist, self.states_to_explain, valid_dict=valid_dict)
            for C in F_not_i(self.F)
        }
        # Convert defaultdicts to regular dicts for pickling
        self.pi_Cs = {k: dict(v) for k, v in self.pi_Cs.items()}
        return self.pi_Cs

    def compute_v_Cs(self, valid_dict=None):
        if self.state_dist is None:
            raise ValueError('State distribution must be calculated before computing v_Cs.')

        self._agent_call('get_value_table', valid_dict=valid_dict)
        self.v_Cs = {
            tuple(C): self._agent_call('get_v_C', C, self.state_dist, self.states_to_explain)
            for C in F_not_i(self.F)
        }
        return self.v_Cs

    def compute_characteristics(self, characteristic_modes, num_rolls, multi_process=False, num_p=1, valid_dict=None):
        if self.pi_Cs is None and any(mode in ['local_sverl', 'global_sverl', 'shapley_on_policy', 'fast_local_sverl'] for mode in characteristic_modes):
            self.compute_pi_Cs(valid_dict=valid_dict)
        if self.v_Cs is None and 'shapley_on_value' in characteristic_modes:
            self.compute_v_Cs()

        values = {}
        for mode in characteristic_modes:
            if mode == 'local_sverl':
                values[mode] = self.characteristics.local_sverl_C_values(
                    num_rolls=num_rolls, pi_Cs=self.pi_Cs, multi_process=multi_process, num_p=num_p)
            elif mode == 'fast_local_sverl':
                values[mode] = self.characteristics.fast_local_sverl_C_values(
                    pi_Cs=self.pi_Cs, num_rolls=num_rolls, valid_dict=valid_dict,
                    multi_process=multi_process, num_p=num_p)
            elif mode == 'global_sverl':
                values[mode] = self.characteristics.global_sverl_C_values(
                    num_rolls=num_rolls, pi_Cs=self.pi_Cs, multi_process=multi_process, num_p=num_p)
            elif mode == 'shapley_on_policy':
                values[mode] = self.characteristics.shapley_on_policy(
                    pi_Cs=self.pi_Cs, multi_process=multi_process, num_p=num_p)
            elif mode == 'shapley_on_value':
                values[mode] = self.characteristics.shapley_on_value(
                    v_Cs=self.v_Cs, multi_process=multi_process, num_p=num_p)
            else:
                raise ValueError(f'Unknown characteristic mode: {mode}')
        return values

    def run_values(self, characteristic_values_map, methods=('shapley', 'banzhaf'), normalized=True):
        results = {}
        if 'shapley' in methods:
            shapley = Shapley(self.states_to_explain)
            results['shapley'] = {
                name: shapley.run(characteristic_values)
                for name, characteristic_values in characteristic_values_map.items()
            }
        if 'banzhaf' in methods:
            banzhaf = Banzhaf(self.states_to_explain)
            results['banzhaf'] = {
                name: banzhaf.run(characteristic_values, normalized=normalized)
                for name, characteristic_values in characteristic_values_map.items()
            }
        if 'nucleolus' in methods:
            nucleolus = Nucleolus(self.states_to_explain)
            results['nucleolus'] = {
                name: nucleolus.run(characteristic_values)
                for name, characteristic_values in characteristic_values_map.items()
            }
        return results

    def explain(self, characteristic_modes, num_rolls, multi_process=False, num_p=1, methods=('shapley', 'banzhaf'), normalized=True):
        characteristic_values = self.compute_characteristics(characteristic_modes, num_rolls, multi_process, num_p)
        return self.run_values(characteristic_values, methods=methods, normalized=normalized)
