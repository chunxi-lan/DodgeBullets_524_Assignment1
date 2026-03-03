import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.3, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        # State discretization parameters (increased precision)
        self.state_bins = [30, 30, 2]  # Discretization bins for plane x-coordinate, cannon angle, and in-range status
        
        # Q-table initialization
        self.q_table = np.zeros(tuple(self.state_bins) + (action_dim,))
        
        # Hyperparameters (tuned)
        self.learning_rate = learning_rate  # Increased learning rate
        self.discount_factor = discount_factor  # Increased discount factor, focus more on long-term rewards
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # Slow down epsilon decay, maintain exploration longer
        self.epsilon_min = epsilon_min
        
        # Action space
        self.action_dim = action_dim
    
    def discretize_state(self, state):
        # Discretize continuous state
        plane_x_norm, angle_norm, in_range = state
        
        # Discretize plane x-coordinate
        plane_x_bin = min(int(plane_x_norm * self.state_bins[0]), self.state_bins[0] - 1)
        
        # Discretize cannon angle (map -1 to 1 to 0 to state_bins[1]-1)
        angle_bin = min(int((angle_norm + 1) / 2 * self.state_bins[1]), self.state_bins[1] - 1)
        
        # Discretize in-range status
        in_range_bin = int(in_range)
        
        return (plane_x_bin, angle_bin, in_range_bin)
    
    def act(self, state, eval_mode=False):
        # Epsilon-greedy strategy for action selection
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        # Q-learning update rule
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Calculate TD target
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[discrete_next_state])
        
        # Update Q-value
        self.q_table[discrete_state + (action,)] += self.learning_rate * (target - self.q_table[discrete_state + (action,)])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        # Save Q-table
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filename):
        # Load Q-table
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
