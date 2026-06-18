import sys
sys.path.insert(0, '../')
from q_agent_2 import Agent
from tic_tac_toe.tic_tac_toe import TTT
from taxi.taxi_wrap import FactoredState
from utils import train, get_state_dist, F_not_i, tqdm_label, value_iteration, find_states_taxi
from characteristics import Characteristics
import numpy as np
import inspect

class GameComputation:
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

    def _agent_get_pi_C(self, C, valid_dict=None):
        sig = inspect.signature(self.agent.get_pi_C)

        if "valid_dict" in sig.parameters:
            return self.agent.get_pi_C(C, self.state_dist, self.states_to_explain, valid_dict=valid_dict)
        return self.agent.get_pi_C(C, self.state_dist, self.states_to_explain)
    
    def _agent_get_v_C(self, C, valid_dict=None):
        sig = inspect.signature(self.agent.get_v_C)

        if "valid_dict" in sig.parameters:
            return self.agent.get_v_C(C, self.state_dist, self.states_to_explain, valid_dict=valid_dict)
        return self.agent.get_v_C(C, self.state_dist, self.states_to_explain)
    
    def _agent_get_value_table(self, C, valid_dict=None):
        sig = inspect.signature(self.agent.get_value_table)

        if "valid_dict" in sig.parameters:
            return self.agent.get_value_table(valid_dict=valid_dict)
        return self.agent.get_value_table()

    def compute_state_dist(self, sample_size=1e6, agent=None, env=None):
        """Approximiert die Zustandsverteilung (wie utils.get_state_dist)."""
        from utils import get_state_dist

        ag = agent or self.agent
        ev = env or self.env
        if ag is None or ev is None:
            raise ValueError("compute_state_dist: agent and environment should have been set.")
        self.state_dist = get_state_dist(ag, ev, int(sample_size))
        return self.state_dist
    
    def get_value_table_for_shapley(self, agent=None, env=None):
        ag = agent or self.agent
        ev = env or self.env
        if ag is None or ev is None:
            raise ValueError("get_value_table_for_shapley: agent and environment should have been set.")
        
        vd = self.valid_dict if self.valid_dict is not None else getattr(
            self.env, "valid_dict", None
        )

        self._agent_get_value_table(vd)
    
    def compute_pi_Cs(self, valid_dict=None):
        """Compute policy characteristic functions pi_C for all coalitions C."""
        if self.agent is None or self.env is None or self.states_to_explain is None:
            self.pi_Cs = {}
            return self.pi_Cs
        
        if getattr(self, "state_dist", None) is None:
            raise ValueError("state_dist missing: call compute_state_dist first.")
        
        vd = valid_dict if valid_dict is not None else getattr(
            self.env, "valid_dict", None
        )

        F = np.arange(self.env.state_dim)
        
        self.pi_Cs = {
            tuple(C): self._agent_get_pi_C(C, valid_dict=vd)
            for C in tqdm_label(F_not_i(F), "    Calculating all pi_C")
        }
        self.pi_Cs = {k: dict(v) for k, v in self.pi_Cs.items()}
        return self.pi_Cs
    
    def compute_v_Cs(self, valid_dict=None):
        """Compute partially observed value tables v_C for all coalitions C."""
        if self.agent is None or self.env is None or self.states_to_explain is None:
            self.v_Cs = {}
            return self.v_Cs
        
        if getattr(self, "state_dist", None) is None:
            raise ValueError("state_dist missing: call compute_state_dist first.")
        
        if not hasattr(self.agent, "value_table"):
            raise ValueError("agent.value_table missing: call get_value_table first.")
        
        vd = valid_dict if valid_dict is not None else getattr(
            self.env, "valid_dict", None
        )

        F = np.arange(self.env.state_dim)
        self.v_Cs = {
            tuple(C): self._agent_get_v_C(C, valid_dict=vd)
            for C in tqdm_label(F_not_i(F), "    Calculating all v_C")
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