import copy
import numpy as np


class Battleship:
    """
    Battleship als MDP: jede Zelle ist ein Merkmal der Beobachtung (state_dim = rows * cols).
    Standard: 5 x 8 = 40 Merkmale.
    """

    NUM_ROWS = 5
    NUM_COLS = 8
    NUM_FEATURES = NUM_ROWS * NUM_COLS  # 40

    def __init__(self, rows=None, cols=None, ship_sizes=None, seed=None):
        rows = self.NUM_ROWS if rows is None else rows
        cols = self.NUM_COLS if cols is None else cols

        self.rows = int(rows)
        self.cols = int(cols)
        self.state_dim = self.rows * self.cols
        self.num_actions = self.state_dim
        self.max_steps = self.state_dim

        self.rng = np.random.default_rng(seed)

        if ship_sizes is None:
            if self.rows >= 5 and self.cols >= 5:
                self.ship_sizes = [3, 2, 2, 1]
            elif max(self.rows, self.cols) >= 3:
                self.ship_sizes = [2, 1, 1]
            else:
                self.ship_sizes = [min(2, self.rows, self.cols)]
        else:
            self.ship_sizes = list(ship_sizes)

        self.total_hits_needed = int(sum(self.ship_sizes))
        if self.total_hits_needed > self.rows * self.cols:
            raise ValueError("Summe der Schiffslängen darf die Zellenanzahl nicht überschreiten.")

        self.opponent_grid = None
        self.player_grid = None
        self.steps = 0
        self.hits = 0

    def _action_to_coords(self, action):
        return divmod(int(action), self.cols)

    def _place_ship(self, grid, size):
        max_attempts = 5000
        for _ in range(max_attempts):
            horizontal = self.rng.random() < 0.5
            if horizontal:
                row = int(self.rng.integers(0, self.rows))
                col = int(self.rng.integers(0, self.cols - size + 1))
                positions = [(row, col + i) for i in range(size)]
            else:
                row = int(self.rng.integers(0, self.rows - size + 1))
                col = int(self.rng.integers(0, self.cols))
                positions = [(row + i, col) for i in range(size)]

            if all(grid[r, c] == 0 for r, c in positions):
                for r, c in positions:
                    grid[r, c] = 1
                return
        raise RuntimeError("Schiffe konnten nicht platziert werden; Raster zu klein oder zu viele Schiffe.")

    def get_valid_actions(self):
        return np.where(self.player_grid.flatten() == 0)[0]

    def _info(self):
        return {"valid_actions": self.get_valid_actions(), "result": None}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.opponent_grid = np.zeros((self.rows, self.cols), dtype=int)
        for size in self.ship_sizes:
            self._place_ship(self.opponent_grid, int(size))

        self.player_grid = np.zeros((self.rows, self.cols), dtype=int)
        self.steps = 0
        self.hits = 0

        obs = self.player_grid.flatten().astype(np.float64)
        return obs, self._info()

    def step(self, action):
        row, col = self._action_to_coords(action)

        if self.player_grid[row, col] != 0:
            obs = self.player_grid.flatten().astype(np.float64)
            return obs, -0.5, False, False, self._info()

        self.steps += 1

        if self.opponent_grid[row, col] == 1:
            self.player_grid[row, col] = 1
            self.hits += 1
            reward = 1.0
        else:
            self.player_grid[row, col] = -1
            reward = -0.1

        terminated = False
        truncated = False
        result = None

        if self.hits == self.total_hits_needed:
            terminated = True
            reward = 10.0
            result = "win"
        elif self.steps >= self.max_steps:
            terminated = True
            reward = -5.0
            result = "loss"

        obs = self.player_grid.flatten().astype(np.float64)
        info = {"valid_actions": self.get_valid_actions(), "result": result}
        return obs, reward, terminated, truncated, info

    def __deepcopy__(self, memo):
        cls = self.__class__
        out = cls.__new__(cls)
        memo[id(self)] = out
        out.rows = self.rows
        out.cols = self.cols
        out.state_dim = self.state_dim
        out.num_actions = self.num_actions
        out.max_steps = self.max_steps
        out.ship_sizes = list(self.ship_sizes)
        out.total_hits_needed = self.total_hits_needed
        out.rng = copy.deepcopy(self.rng, memo)
        out.opponent_grid = copy.deepcopy(self.opponent_grid, memo)
        out.player_grid = copy.deepcopy(self.player_grid, memo)
        out.steps = self.steps
        out.hits = self.hits
        return out
