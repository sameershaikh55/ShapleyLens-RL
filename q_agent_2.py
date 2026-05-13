import numpy as np
from collections import defaultdict


class Agent:
    """
    Q-Learning Agent mit generischer Zustandsrepräsentation.
    Funktioniert für unterschiedliche Environments.
    """

    def __init__(self, state_dim, num_actions):

        self.state_dim = state_dim
        self.num_actions = num_actions

        # Lernparameter
        self.alpha = 0.1          # Lernrate
        self.gamma = 0.95         # Diskontfaktor

        # Exploration (Epsilon-Greedy)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Q-Tabelle
        self.Q_table = defaultdict(lambda: np.zeros(self.num_actions))

    # -------------------------------------------------
    # Zustandsrepräsentation (WICHTIG)
    # -------------------------------------------------
    def _state_to_key(self, state):
        """
        Wandelt beliebige Zustände in ein stabiles Tupel um.
        Unterstützt:
        - int
        - tuple
        - list
        - numpy array
        """

        # Einzelner Integer-Zustand
        if isinstance(state, (int, np.integer)):
            return (int(state),)

        # Tuple bleibt gleich
        if isinstance(state, tuple):
            return tuple(state)

        # Liste → Tuple
        if isinstance(state, list):
            return tuple(state)

        # NumPy Array → flatten → Tuple
        if isinstance(state, np.ndarray):
            return tuple(state.astype(int).flatten())

        # Fallback (für Sicherheit)
        return tuple(np.array(state, dtype=int).flatten())

    # -------------------------------------------------
    # Aktionsauswahl
    # -------------------------------------------------
    def choose_action(self, state, info):
        """
        Wählt eine Aktion mittels Epsilon-Greedy Strategie.
        """

        state_key = self._state_to_key(state)
        valid_actions = info["valid_actions"]

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)

        # Exploitation
        q_values = self.Q_table[state_key]
        return valid_actions[np.argmax(q_values[valid_actions])]

    # -------------------------------------------------
    # Q-Learning Update
    # -------------------------------------------------
    def update(self, state, action, reward, new_state, done, info):
        """
        Aktualisiert die Q-Tabelle gemäß der Q-Learning Formel.
        """

        state_key = self._state_to_key(state)
        new_state_key = self._state_to_key(new_state)

        # Zielwert berechnen
        if done:
            q_max = 0
        else:
            q_max = self.Q_table[new_state_key][info["valid_actions"]].max()

        # Q-Learning Formel
        self.Q_table[state_key][action] += self.alpha * (
            reward + self.gamma * q_max - self.Q_table[state_key][action]
        )

        # Exploration reduzieren
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
