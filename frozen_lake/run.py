import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from q_agent_1 import Agent
from frozen_lake.frozen_lake_wrap import FrozenLakeWrap
from utils import get_state_dist, value_iteration
from game_computation import GameComputation

# Configuration
GAMMA = 0.99
NUM_ROLLS = 10000

def frozen_lake_init():
# Initialize environment
    env = FrozenLakeWrap(is_slippery=True)
    
    # Initialize agent
    agent = Agent(env.state_dim, env.num_actions, epsilon=0.0, gamma=GAMMA, alpha=0.2)
    
    # Compute Q-table and policy using value iteration
    agent.Q_table, agent.policy = value_iteration(env, gamma=GAMMA)
    
    # Identify non-terminal states (states that are in the policy)
    # Terminal states (goal and holes) won't have entries in agent.policy for FrozenLake
    non_terminal_states = list(agent.policy.keys())
    
    # Select states to explain from non-terminal states
    # Use the first 6 non-terminal states or all if fewer than 6
    states_to_explain = np.array(non_terminal_states[:6] if len(non_terminal_states) >= 6 else non_terminal_states)
    return env, agent, states_to_explain

def frozen_lake_get_characteristic_modes():
    return ['local_sverl', 'shapley_on_policy', 'shapley_on_value']

def frozen_lake_run(env=None, agent=None, states_to_explain=None):
    # Get value table from Q-table
    agent.get_value_table()
    
    # Approximate state distribution
    state_dist = get_state_dist(agent, env, sample_size=10000)
    
    # Initialize explainer
    game_compute = GameComputation(env, agent, states_to_explain)
    game_compute.state_dist = state_dist
    
    # Compute partial policies and values
    game_compute.compute_pi_Cs()
    game_compute.get_value_table_for_shapley()
    game_compute.compute_v_Cs()
    
    # Compute characteristic values
    characteristic_modes = frozen_lake_get_characteristic_modes()
    characteristics = game_compute.compute_characteristics(
        characteristic_modes, num_rolls=NUM_ROLLS, multi_process=False, num_p=1)
    return characteristics
