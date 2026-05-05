try:
    import gym  # type: ignore[import]
except ImportError:
    import gymnasium as gym
import numpy as np

# Set environment and agent
class FrozenLakeWrap(gym.ObservationWrapper):
    """
    Wrapper for FrozenLake environment that factorizes the state.
    """

    def __init__(self, is_slippery=True):
        
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode=None)
        super().__init__(env)

        # FrozenLake uses 4x4 grid by default, so state is integer 0-15
        # We'll convert it to 2D coordinates [row, col]
        self._observation_space = gym.spaces.MultiDiscrete([4, 4])

        self.state_dim = self.observation_space.shape[0]
        self.num_actions = self.action_space.n
        self.P = env.unwrapped.P
        self.grid_size = 4
        self.current_state = None

    def decode(self, obs):
        """
        Decode integer state to 2D coordinates [row, col]
        Returns regular Python floats, not numpy floats
        """
        row = int(obs // self.grid_size)
        col = int(obs % self.grid_size)
        return np.array([float(row), float(col)])
    
    def encode(self, state):
        """
        Encode 2D coordinates [row, col] to integer state
        """
        if isinstance(state, np.ndarray):
            state = state.astype(int)
        return state[0] * self.grid_size + state[1]

    def reset(self, start_state=None, **kwargs):
        """
        Reset the environment to a specific state or random state.
        If start_state is provided, set the environment to that state.
        Otherwise, use default reset.
        """
        if start_state is not None:
            # Convert factored state to integer if needed
            if isinstance(start_state, (list, tuple, np.ndarray)):
                state_int = self.encode(start_state)
            else:
                state_int = start_state
            
            # Set the unwrapped environment's state directly
            self.env.unwrapped.s = state_int
            self.current_state = state_int
            obs = self.observation(state_int)
            return obs, {}
        else:
            # Use default reset
            obs, info = self.env.reset(**kwargs)
            self.current_state = obs if isinstance(obs, int) else self.encode(obs)
            return self.observation(self.current_state), info

    def observation(self, obs):
        """
        Convert observation from integer to 2D coordinates
        """
        return self.decode(obs)
