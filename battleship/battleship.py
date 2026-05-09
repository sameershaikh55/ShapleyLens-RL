import numpy as np


class Battleship():
    """
    Battleship environment. Player tries to sink all opponent ships.
    Simple single-player version where opponent ships are pre-placed.
    """
    
    def __init__(self, grid_size=10, max_steps=None):
        """
        Initialize Battleship environment.
        
        Args:
            grid_size: Size of the square grid (default 10x10)
            max_steps: Maximum number of actions before game ends (default None = no limit)
        """
        
        self.grid_size = grid_size
        self.max_steps = max_steps if max_steps else grid_size * grid_size
        
        # Ship sizes in standard battleship
        self.ship_sizes = [5, 4, 3, 3, 2]  # Battleship, Cruiser, Submarine, Destroyer, Patrol Boat
        self.num_ships = len(self.ship_sizes)
        
        # Total number of cells to hit to win
        self.total_hits_needed = sum(self.ship_sizes)
        
        # Number of possible actions (grid coordinates)
        self.num_actions = grid_size * grid_size
        
        # State dimension: flattened grid showing hits/misses/unknown
        self.state_dim = grid_size * grid_size
        
        # Reward dictionary: hit=1, miss=0, game_won=10
        self.reward_dict = {'hit': 1, 'miss': 0, 'win': 10, 'loss': -1}
        
        # Cache for valid actions
        self.valid_dict = ValidDict(grid_size)
        
    def _action_to_coords(self, action):
        """Convert action (0-99) to grid coordinates (row, col)."""
        return divmod(action, self.grid_size)
    
    def _coords_to_action(self, row, col):
        """Convert grid coordinates to action index."""
        return row * self.grid_size + col
    
    def _place_ship(self, grid, ship_size, attempts=0):
        """
        Randomly place a ship on the grid.
        Returns True if successful, False if can't place after max attempts.
        """
        if attempts > 100:
            return False
        
        # Random orientation: 0=horizontal, 1=vertical
        horizontal = np.random.rand() < 0.5
        
        if horizontal:
            row = np.random.randint(0, self.grid_size)
            col = np.random.randint(0, self.grid_size - ship_size + 1)
            positions = [(row, col + i) for i in range(ship_size)]
        else:
            row = np.random.randint(0, self.grid_size - ship_size + 1)
            col = np.random.randint(0, self.grid_size)
            positions = [(row + i, col) for i in range(ship_size)]
        
        # Check if all positions are empty
        if all(grid[r, c] == 0 for r, c in positions):
            for r, c in positions:
                grid[r, c] = 1
            return True
        else:
            return self._place_ship(grid, ship_size, attempts + 1)
    
    def reset(self, start_state=None):
        """
        Reset the environment for a new game.
        """
        
        if start_state is None:
            # Create opponent's ship grid
            self.opponent_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
            
            # Place all ships
            for ship_size in self.ship_sizes:
                self._place_ship(self.opponent_grid, ship_size)
            
            # Player's knowledge grid: 0=unknown, 1=hit, -1=miss
            self.player_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
            
        else:
            self.player_grid = start_state.copy().reshape(self.grid_size, self.grid_size)
            # Opponent grid not needed for continuing game
            self.opponent_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        self.steps_taken = 0
        self.hits_count = 0
        
        valid_actions = self._get_valid_actions()
        
        return self.player_grid.flatten(), {'valid_actions': valid_actions}
    
    def _get_valid_actions(self):
        """Get list of cells that haven't been shot yet."""
        return np.where(self.player_grid.flatten() == 0)[0]
    
    def step(self, action):
        """
        Take an action (shoot at a grid location).
        Returns: (state, reward, done, truncated, info)
        """
        
        row, col = self._action_to_coords(action)
        
        # Check if this cell was already shot
        if self.player_grid[row, col] != 0:
            reward = self.reward_dict['loss']
            return self.player_grid.flatten(), reward, True, False, {'valid_actions': []}
        
        self.steps_taken += 1
        
        # Check if it's a hit
        if self.opponent_grid[row, col] == 1:
            self.player_grid[row, col] = 1
            self.hits_count += 1
            reward = self.reward_dict['hit']
        else:
            self.player_grid[row, col] = -1
            reward = self.reward_dict['miss']
        
        done = False
        
        # Check if all ships are sunk (won)
        if self.hits_count == self.total_hits_needed:
            reward += self.reward_dict['win']
            done = True
        
        # Check if max steps exceeded (lost)
        elif self.steps_taken >= self.max_steps:
            reward = self.reward_dict['loss']
            done = True
        
        valid_actions = self._get_valid_actions()
        
        return self.player_grid.flatten(), reward, done, False, {'valid_actions': valid_actions}
    
    def valid_actions(self, state):
        """Get valid actions for a given state."""
        return np.where(state == 0)[0]


class ValidDict(dict, Battleship):
    """
    Caches valid actions for each state to improve performance.
    """
    
    def __init__(self, grid_size):
        """Initialize ValidDict with grid_size."""
        dict.__init__(self)
        self.grid_size = grid_size
    
    def __missing__(self, key):
        val = self.valid_actions(np.frombuffer(key, dtype=np.int_).reshape(self.grid_size, self.grid_size).flatten())
        self.__setitem__(key, val)
        return val
