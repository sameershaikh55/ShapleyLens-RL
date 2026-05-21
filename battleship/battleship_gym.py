import copy

import numpy as np

try:
    import gym  # type: ignore[import]
except ImportError:
    import gymnasium as gym
from gymnasium import spaces

from battleship import Battleship


class BattleshipEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 2,
    }

    NUM_ROWS = Battleship.NUM_ROWS
    NUM_COLS = Battleship.NUM_COLS
    NUM_FEATURES = Battleship.NUM_FEATURES

    _CELL_COLORS = {
        0: np.array([30, 90, 180], dtype=np.uint8),
        1: np.array([220, 50, 50], dtype=np.uint8),
        -1: np.array([200, 210, 220], dtype=np.uint8),
    }
    _OPPONENT_SHIP_COLOR = np.array([60, 60, 60], dtype=np.uint8)
    _OPPONENT_WATER_COLOR = np.array([20, 60, 120], dtype=np.uint8)

    def __init__(
        self,
        rows=None,
        cols=None,
        ship_sizes=None,
        seed=None,
        render_mode=None,
        reveal_opponent=False,
        cell_size=32,
    ):
        super().__init__()

        self.game = Battleship(rows=rows, cols=cols, ship_sizes=ship_sizes, seed=seed)
        self.render_mode = render_mode
        self.reveal_opponent = bool(reveal_opponent)
        self.cell_size = int(cell_size)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}"
            )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.game.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.game.num_actions)

    @property
    def rows(self):
        return self.game.rows

    @property
    def cols(self):
        return self.game.cols

    @property
    def state_dim(self):
        return self.game.state_dim

    @property
    def num_actions(self):
        return self.game.num_actions

    @property
    def max_steps(self):
        return self.game.max_steps

    @property
    def ship_sizes(self):
        return self.game.ship_sizes

    @property
    def total_hits_needed(self):
        return self.game.total_hits_needed

    @property
    def opponent_grid(self):
        return self.game.opponent_grid

    @property
    def player_grid(self):
        return self.game.player_grid

    @property
    def steps(self):
        return self.game.steps

    @property
    def hits(self):
        return self.game.hits

    def get_valid_actions(self):
        return self.game.get_valid_actions()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.game.reset(seed=seed, options=options)

    def step(self, action):
        return self.game.step(action)

    def _cell_char(self, value):
        if value == 0:
            return "."
        if value == 1:
            return "X"
        return "o"

    def _render_board_lines(self, grid, title):
        lines = [title]
        col_header = "   " + " ".join(str(c) for c in range(self.cols))
        lines.append(col_header)
        for row_idx, row in enumerate(grid):
            cells = " ".join(self._cell_char(int(value)) for value in row)
            lines.append(f"{row_idx}  {cells}")
        return lines

    def _render_opponent_lines(self):
        lines = ["Gegner (Debug):"]
        col_header = "   " + " ".join(str(c) for c in range(self.cols))
        lines.append(col_header)
        for row_idx, row in enumerate(self.opponent_grid):
            cells = " ".join("#" if int(value) == 1 else "~" for value in row)
            lines.append(f"{row_idx}  {cells}")
        return lines

    def _render_text(self):
        lines = [
            f"Schritt {self.steps} | Treffer {self.hits}/{self.total_hits_needed}",
        ]
        lines.extend(self._render_board_lines(self.player_grid, "Spielfeld:"))
        if self.reveal_opponent:
            lines.append("")
            lines.extend(self._render_opponent_lines())
        return "\n".join(lines)

    def _render_rgb_board(self, grid, *, opponent_view=False):
        cell = self.cell_size
        height = self.rows * cell
        width = self.cols * cell
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):
                value = int(grid[row, col])
                if opponent_view:
                    color = (
                        self._OPPONENT_SHIP_COLOR
                        if value == 1
                        else self._OPPONENT_WATER_COLOR
                    )
                else:
                    color = self._CELL_COLORS[value]
                y0, y1 = row * cell, (row + 1) * cell
                x0, x1 = col * cell, (col + 1) * cell
                image[y0:y1, x0:x1] = color

        return image

    def _render_rgb_array(self):
        player_image = self._render_rgb_board(self.player_grid)
        if not self.reveal_opponent:
            return player_image

        gap = np.full((player_image.shape[0], 8, 3), 255, dtype=np.uint8)
        opponent_image = self._render_rgb_board(self.opponent_grid, opponent_view=True)
        return np.concatenate([player_image, gap, opponent_image], axis=1)

    def _render_frame(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return self._render_text()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "render() called without render_mode set. Set render_mode in __init__ or gym.make()."
            )
            return None

        frame = self._render_frame()
        if self.render_mode == "human":
            print(frame)
            return None
        return frame

    def close(self):
        pass

    def __deepcopy__(self, memo):
        cls = self.__class__
        out = cls.__new__(cls)
        memo[id(self)] = out
        out.game = copy.deepcopy(self.game, memo)
        out.render_mode = self.render_mode
        out.reveal_opponent = self.reveal_opponent
        out.cell_size = self.cell_size
        out.observation_space = self.observation_space
        out.action_space = self.action_space
        return out
