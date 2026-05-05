import sys
sys.path.insert(0, '../')

import pickle
import numpy as np
from q_agent_1 import Agent
from frozen_lake_wrap import FrozenLakeWrap
from utils import get_state_dist, value_iteration
from explainer import Explainer

# Configuration
GAMMA = 0.99
NUM_ROLLS = 10000

if __name__ == '__main__':
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
    
    # Get value table from Q-table
    agent.get_value_table()
    
    # Approximate state distribution
    state_dist = get_state_dist(agent, env, sample_size=10000)
    
    # Initialize explainer
    explainer = Explainer(env, agent, states_to_explain)
    explainer.state_dist = state_dist
    
    # Compute partial policies and values
    explainer.compute_pi_Cs()
    explainer.compute_v_Cs()
    
    # Compute characteristic values
    characteristic_modes = ['local_sverl', 'shapley_on_policy', 'shapley_on_value']
    characteristics = explainer.compute_characteristics(
        characteristic_modes, num_rolls=NUM_ROLLS, multi_process=False, num_p=1)
    
    # Compute Shapley and Banzhaf values
    results = explainer.run_values(characteristics, methods=('shapley', 'banzhaf'), normalized=True)
    
    # Save results
    for method, method_results in results.items():
        for name, values in method_results.items():
            print(f"\n=== {method.capitalize()} ({name}) ===")
            print(values)
            with open(f'{method}_{name}.pkl', 'wb') as file:
                pickle.dump(values, file)
