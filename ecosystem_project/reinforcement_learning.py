"""
Reinforcement Learning implementation for agent decision making in the ecosystem simulation.
This provides a simple Q-learning implementation for agents to learn behaviors.
"""

import numpy as np
from collections import defaultdict

class QLearning:
    """A simple Q-learning implementation for reinforcement learning."""
    
    def __init__(self, model, state_features=3, action_space=4, learning_rate=0.1, discount_factor=0.9):
        """Initialize Q-learning agent.
        
        Args:
            model: Model instance
            state_features: Number of features that define a state
            action_space: Number of possible actions
            learning_rate: Rate at which Q-values are updated
            discount_factor: How much future rewards are valued vs immediate ones
        """
        self.model = model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = 0.3
        self.min_exploration_rate = 0.05
        self.exploration_decay = 0.995
        
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        
        self.state_features = state_features
        self.action_space = action_space
        
        self.rewards = []
        
    def discretize_state(self, state_dict):
        """Convert a dictionary of continuous state values to a discrete state key.
        
        Args:
            state_dict: Dictionary with continuous state values
            
        Returns:
            A tuple that can be used as a dictionary key
        """
        discrete_state = []
        
        for key, value in sorted(state_dict.items()):
            if key == 'energy':
                if value < 0.3:
                    discrete_state.append(0)
                elif value < 0.7:
                    discrete_state.append(1)
                else:
                    discrete_state.append(2)
            elif key == 'danger':
                discrete_state.append(int(min(1, value)))
            elif key == 'food':
                if value < 0.1:
                    discrete_state.append(0)
                elif value < 0.5:
                    discrete_state.append(1)
                else:
                    discrete_state.append(2)
            else:
                discrete_state.append(min(2, int(value * 3)))
                
        return tuple(discrete_state)
    
    def select_action(self, state_dict):
        """Select an action based on current state using epsilon-greedy policy.
        
        Args:
            state_dict: Dictionary with state information
            
        Returns:
            Selected action index
        """
        state = self.discretize_state(state_dict)
        
        if self.model.random.random() < self.exploration_rate:
            return self.model.random.randint(0, self.action_space - 1)
        
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state_dict, action, reward, next_state_dict):
        """Update Q-value for a state-action pair using the Q-learning formula.
        
        Args:
            state_dict: State before action
            action: Action taken
            reward: Reward received
            next_state_dict: State after action
        """
        state = self.discretize_state(state_dict)
        next_state = self.discretize_state(next_state_dict)
        
        current_q = self.q_table[state][action]
        
        max_future_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q)
        
        self.q_table[state][action] = new_q
        
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        self.rewards.append(reward)
        if len(self.rewards) > 100:
            self.rewards.pop(0)
    
    def get_average_reward(self):
        """Get the average reward over recent experiences.
        
        Returns:
            Average reward or 0 if no rewards yet
        """
        if not self.rewards:
            return 0
        return sum(self.rewards) / len(self.rewards)
    
    def mutate(self, mutation_rate=0.1):
        """Create a mutated version of this Q-learning agent.
        
        Args:
            mutation_rate: Probability of mutation for each Q-value
            
        Returns:
            New QLearning instance with mutated Q-values
        """
        new_agent = QLearning(
            self.model, 
            self.state_features, 
            self.action_space,
            self.learning_rate,
            self.discount_factor
        )
        
        for state, actions in self.q_table.items():
            new_agent.q_table[state] = actions.copy()
            
            if self.model.random.random() < mutation_rate:
                noise = self.model.random.randn(self.action_space) * 0.2
                new_agent.q_table[state] += noise
        
        return new_agent
    
    def crossover(self, other_agent):
        """Create a new Q-learning agent by combining two parents.
        
        Args:
            other_agent: Another QLearning instance to combine with
            
        Returns:
            New QLearning instance with mixed Q-values
        """
        child = QLearning(
            self.model, 
            self.state_features, 
            self.action_space,
            self.learning_rate,
            self.discount_factor
        )
        
        all_states = set(list(self.q_table.keys()) + list(other_agent.q_table.keys()))
        
        for state in all_states:
            if state in self.q_table and state in other_agent.q_table:
                if self.model.random.random() < 0.5:
                    child.q_table[state] = self.q_table[state].copy()
                else:
                    child.q_table[state] = other_agent.q_table[state].copy()
            elif state in self.q_table:
                child.q_table[state] = self.q_table[state].copy()
            else:
                child.q_table[state] = other_agent.q_table[state].copy()
        
        return child
        
    def interpret_action(self, action_index):
        """Convert action index to a more usable form.
        
        Args:
            action_index: Index of the selected action
            
        Returns:
            Dictionary with action details
        """
        if action_index == 0:
            return {'direction': 'north', 'dx': 0, 'dy': -1}
        elif action_index == 1:
            return {'direction': 'east', 'dx': 1, 'dy': 0}
        elif action_index == 2:
            return {'direction': 'south', 'dx': 0, 'dy': 1}
        elif action_index == 3:
            return {'direction': 'west', 'dx': -1, 'dy': 0}
        else:
            return {'direction': 'none', 'dx': 0, 'dy': 0}
