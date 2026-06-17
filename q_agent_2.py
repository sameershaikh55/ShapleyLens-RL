import copy
from collections import defaultdict

import numpy as np

from utils import mask_state


class Agent:
    """
    Q-Learning Agent mit generischer Zustandsrepräsentation.
    Unterstützt Training (valid_actions, Epsilon-Decay) und Shapley/Explainer-Methoden.
    """

    def __init__(self, state_dim, num_actions, epsilon, gamma, alpha):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.actions = np.arange(self.num_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.Q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def _state_to_key(self, state):
        if isinstance(state, (int, np.integer)):
            return (int(state),)
        if isinstance(state, tuple):
            return tuple(state)
        if isinstance(state, list):
            return tuple(state)
        if isinstance(state, np.ndarray):
            return tuple(state.astype(int).flatten())
        return tuple(np.array(state, dtype=int).flatten())

    def choose_action(self, state, info, exp=True):
        state_key = self._state_to_key(state)
        valid_actions = info["valid_actions"]

        if np.random.rand() < self.epsilon and exp:
            return np.random.choice(valid_actions)

        q_values = self.Q_table[state_key][valid_actions]
        return np.random.choice(valid_actions[q_values == q_values.max()])

    def update(self, state, action, new_state, reward, done, info):
        state_key = self._state_to_key(state)
        new_state_key = self._state_to_key(new_state)

        if done:
            q_max = 0
        else:
            q_max = self.Q_table[new_state_key][info["valid_actions"]].max()

        td_error = reward + self.gamma * q_max - self.Q_table[state_key][action]
        self.Q_table[state_key][action] += self.alpha * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_policy(self, valid_dict=None):
        if valid_dict is None:
            self.policy = defaultdict(lambda: np.full(self.num_actions, 1 / self.num_actions))
            for state, q_values in self.Q_table.items():
                q_values = q_values.round(1)
                self.policy[state] = (q_values == q_values.max()).astype(float)
                self.policy[state] /= self.policy[state].sum()
            return

        self.policy = PolicyDict()
        self.policy.valid_dict = valid_dict
        self.policy.num_actions = self.num_actions

        for state, q_values in self.Q_table.items():
            actions = valid_dict[np.array(state).tobytes()]
            masked_q = q_values[actions].round(2)
            self.policy[state][actions] = masked_q == masked_q.max()
            self.policy[state] /= self.policy[state].sum()

    def get_value_table(self, valid_dict=None):
        if valid_dict is None:
            self.value_table = {
                state: q_values.max() for state, q_values in self.Q_table.items()
            }
            return

        self.value_table = {
            state: q_values[valid_dict[np.array(state).tobytes()]].max()
            for state, q_values in self.Q_table.items()
        }

    def get_pi_C(self, C, state_dist, states_to_explain, valid_dict=None):
        if len(C) == self.state_dim:
            return copy.deepcopy(self.policy)

        policy_keys = list(self.policy.keys())
        all_states = np.array(policy_keys)
        mask_states = mask_state(states_to_explain, self.state_dim, C)
        mask_all_states = mask_state(all_states, self.state_dim, C)
        state_dist_full = np.array([state_dist.get(k, 0.0) for k in policy_keys]) + 1e-16
        policy_matrix = np.array([self.policy[k] for k in policy_keys])

        if valid_dict is None:
            pi_C = defaultdict(lambda: np.full(self.num_actions, 1 / self.num_actions))
            temp_pi_C = {}

            for m_state in np.unique(mask_states, axis=0):
                ind = (mask_all_states == m_state).all(axis=1)
                state_dist_cond = state_dist_full[ind] / state_dist_full[ind].sum()
                temp_pi_C[tuple(m_state)] = (policy_matrix[ind] * state_dist_cond[:, None]).sum(axis=0)

            for state, m_state in zip(states_to_explain, mask_states):
                pi_C[tuple(state)] = temp_pi_C[tuple(m_state)]

            return pi_C

        pi_C = PolicyDict()
        pi_C.valid_dict = valid_dict
        pi_C.num_actions = self.num_actions
        temp_pi_C = {}

        for m_state in np.unique(mask_states, axis=0):
            ind = (mask_all_states == m_state).all(axis=1)
            state_dist_cond = state_dist_full[ind] / state_dist_full[ind].sum()
            temp_pi_C[tuple(m_state)] = (policy_matrix[ind] * state_dist_cond[:, None]).sum(axis=0)

        for state, m_state in zip(states_to_explain, mask_states):
            actions = valid_dict[state.tobytes()]
            pi_C[tuple(state)][actions] = temp_pi_C[tuple(m_state)][actions]
            pi_C[tuple(state)] /= pi_C[tuple(state)].sum()

        return pi_C

    def get_v_C(self, C, state_dist, states_to_explain):
        policy_keys = list(self.policy.keys())
        all_states = np.array(policy_keys)
        mask_states = mask_state(states_to_explain, self.state_dim, C)
        mask_all_states = mask_state(all_states, self.state_dim, C)

        state_dist_full = np.array([state_dist.get(k, 0.0) for k in policy_keys]) + 1e-16
        values = np.array([self.value_table.get(k, 0.0) for k in policy_keys])

        v_C = {}
        temp_v_C = {}

        for m_state in np.unique(mask_states, axis=0):
            ind = (mask_all_states == m_state).all(axis=1)
            state_dist_cond = state_dist_full[ind] / state_dist_full[ind].sum()
            temp_v_C[tuple(m_state)] = (values[ind] * state_dist_cond).sum()

        for state, m_state in zip(states_to_explain, mask_states):
            v_C[tuple(state)] = temp_v_C[tuple(m_state)]

        return v_C


class PolicyDict(dict):
    """Policy dict that defaults to uniform over valid actions only."""

    def __missing__(self, key):
        valid_actions = self.valid_dict[np.array(key).tobytes()]
        val = np.zeros(self.num_actions)
        val[valid_actions] = 1 / len(valid_actions)
        self.__setitem__(key, val)
        return val
