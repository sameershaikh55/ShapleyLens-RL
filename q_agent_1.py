import numpy as np
from collections import defaultdict
from utils import mask_state
import copy
    
class Agent:
    """
    Usual Q learning agent, with added functionality for calculting Shapley values.
    """
    
    def __init__(self, state_dim, num_actions, epsilon=0.05, gamma=0.99, alpha=0.2):

        self.num_actions = num_actions
        self.state_dim = state_dim
        self.actions = np.arange(self.num_actions)
        
        # Agent hyper-parameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        self.Q_table = defaultdict(lambda: np.zeros(self.num_actions))
        
    def choose_action(self, state, info, exp=True):
        """
        Chooses action with epsilon greedy policy.
        exp sets whether agent explores or is solely greedy.
        """

        if np.random.rand() < self.epsilon and exp: return np.random.randint(self.num_actions)
        else: 
            q_values = self.Q_table[tuple(state)]
            return np.random.choice(self.actions[q_values == q_values.max()])
        
    def update(self, state, action, new_state, reward, done, info):
        """
        Q learning update.
        """
        
        # Don't want terminal state to ever appear in q_table
        if done: q_max = 0
        else: q_max = self.Q_table[tuple(new_state)].max()
        
        # Usual update.
        td_error = reward + self.gamma * q_max - self.Q_table[tuple(state)][action]
        self.Q_table[tuple(state)][action] += self.alpha * td_error

# ------------------------------------------------------------------ Usual agent stuff finishes.

    def get_policy(self):
        """
        Converts the agent's Q table into a policy table.
        """

        self.policy = defaultdict(lambda: np.full(self.num_actions, 1/self.num_actions))

        for state, q_values in self.Q_table.items():

            # Q values with slightly different values but should be same policy. 
            q_values = q_values.round(1)

            self.policy[state] = (q_values == q_values.max()).astype(float)
            self.policy[state] /= self.policy[state].sum()

    def get_value_table(self):
        """
        Converts the agent's Q table into a value table.
        """

        self.value_table = {state: q_values.max() for state, q_values in self.Q_table.items()}

    def get_pi_C(self, C, state_dist, states_to_explain):
        """
        Calculates pi_C for given states.
        """

        if len(C) == self.state_dim: return copy.deepcopy(self.policy)
        else:

            # Gleiche Reihenfolge wie state_dist (aus policy-Keys in get_state_dist).
            policy_keys = list(self.policy.keys())
            all_states = np.array(policy_keys)
            mask_states = mask_state(states_to_explain, self.state_dim, C)
            mask_all_states = mask_state(all_states, self.state_dim, C)

            state_dist_full = np.array([state_dist.get(k, 0.0) for k in policy_keys]) + 1e-16
            policy_matrix = np.array([self.policy[k] for k in policy_keys])

            pi_C = defaultdict(lambda: np.full(self.num_actions, 1/self.num_actions))
            temp_pi_C = {}

            for m_state in np.unique(mask_states, axis=0):

                ind = (mask_all_states == m_state).all(axis=1)
                state_dist_cond = state_dist_full[ind] / state_dist_full[ind].sum()

                temp_pi_C[tuple(m_state)] = (policy_matrix[ind] * state_dist_cond[:, None]).sum(axis=0)

            # Set partially observed policies for fully observed states using^
            for state, m_state in zip(states_to_explain, mask_states):

                pi_C[tuple(state)] = temp_pi_C[tuple(m_state)] 

            return pi_C
    
    def get_v_C(self, C, state_dist, states_to_explain):
        """
        Calculates the partially observed prediction for state-values table (for Shapley on value function).
        """

        # Gleiche Zustandsmenge/Reihenfolge wie state_dist (policy-Keys), nicht Q_table.keys()
        # (sonst Längen-Mismatch zu state_dist_full bei boolean mask ind).
        policy_keys = list(self.policy.keys())
        all_states = np.array(policy_keys)
        mask_states = mask_state(states_to_explain, self.state_dim, C)
        mask_all_states = mask_state(all_states, self.state_dim, C)

        state_dist_full = np.array([state_dist.get(k, 0.0) for k in policy_keys]) + 1e-16
        values = np.array([self.value_table[k] for k in policy_keys])

        v_C = {}
        temp_v_C = {}

        for m_state in np.unique(mask_states, axis=0):

            ind = (mask_all_states == m_state).all(axis=1)
            state_dist_cond = state_dist_full[ind] / state_dist_full[ind].sum()

            temp_v_C[tuple(m_state)] = (values[ind] * state_dist_cond).sum()

        # Set partially observed values for fully observed states using^
        for state, m_state in zip(states_to_explain, mask_states):

            v_C[tuple(state)] = temp_v_C[tuple(m_state)]
        
        return v_C