
import gymnasium as gym
import numpy as np

MAP_4x4 = [
    ["S", "F", "F", "F"],
    ["F", "H", "F", "H"],
    ["F", "F", "F", "H"],
    ["H", "F", "F", "G"],
]

HOLES = [5, 7, 11, 12]
GOAL = 15


class FrozenLakeWrap:
    def __init__(self, map_name="4x4", is_slippery=True):
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery,
        )

        # Für Compatibility mit utils.value_iteration
        self.unwrapped = self.env.unwrapped
        self.P = self.unwrapped.P

        self.grid_size = 4
        self.n_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.state_dim = 2  # (row, col)

    def reset(self, state=None):
        obs, info = self.env.reset()

        if state is not None:
            s_int = self.encode(state)
            self.unwrapped.s = s_int
            obs = s_int

        return self.decode(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.decode(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def decode(self, s_int):
        return np.array(
            [s_int // self.grid_size, s_int % self.grid_size],
            dtype=int,
        )

    def encode(self, state):
        return int(state[0] * self.grid_size + state[1])

    def feature_names(self):
        return ["row", "col"]