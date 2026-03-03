import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.3, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        # 状态离散化参数（增加精度）
        self.state_bins = [30, 30, 2]  # 飞机x坐标、炮口角度、是否在射程内的离散化区间数（增加精度）
        
        # Q表初始化
        self.q_table = np.zeros(tuple(self.state_bins) + (action_dim,))
        
        # 超参数（调整）
        self.learning_rate = learning_rate  # 增加学习率
        self.discount_factor = discount_factor  # 增加折扣因子，更关注长期奖励
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # 减慢epsilon衰减，保持更长时间的探索
        self.epsilon_min = epsilon_min
        
        # 动作空间
        self.action_dim = action_dim
    
    def discretize_state(self, state):
        # 将连续状态离散化
        plane_x_norm, angle_norm, in_range = state
        
        # 离散化飞机x坐标
        plane_x_bin = min(int(plane_x_norm * self.state_bins[0]), self.state_bins[0] - 1)
        
        # 离散化炮口角度（-1到1映射到0到state_bins[1]-1）
        angle_bin = min(int((angle_norm + 1) / 2 * self.state_bins[1]), self.state_bins[1] - 1)
        
        # 离散化是否在射程内
        in_range_bin = int(in_range)
        
        return (plane_x_bin, angle_bin, in_range_bin)
    
    def act(self, state, eval_mode=False):
        # ε-greedy策略选择动作
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        # Q-learning更新规则
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # 计算TD目标
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[discrete_next_state])
        
        # 更新Q值
        self.q_table[discrete_state + (action,)] += self.learning_rate * (target - self.q_table[discrete_state + (action,)])
        
        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        # 保存Q表
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filename):
        # 加载Q表
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
