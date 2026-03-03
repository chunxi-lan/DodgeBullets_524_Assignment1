import numpy as np
import matplotlib.pyplot as plt
from env import AntiAircraftEnv
from agent import QLearningAgent
import os

# 训练参数
EPISODES = 20000  # 训练2万次
MAX_STEPS = 400  # 步数翻倍，提供更多射击机会

# 创建环境和智能体
env = AntiAircraftEnv()
agent = QLearningAgent(state_dim=env.state_dim, action_dim=env.action_dim)

# 记录奖励
rewards = []
avg_rewards = []

# 训练循环
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    
    for step in range(MAX_STEPS):
        # 选择动作
        action = agent.act(state)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 学习
        agent.learn(state, action, reward, next_state, done)
        
        # 更新状态和奖励
        state = next_state
        total_reward += reward
        
        # 每100个episode渲染一次
        if episode % 100 == 0:
            env.render()
    
    # 记录奖励
    rewards.append(total_reward)
    avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    avg_rewards.append(avg_reward)
    
    # 打印训练信息
    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # 每200个episode保存模型
    if episode % 200 == 0:
        os.makedirs('models', exist_ok=True)
        agent.save(f'models/q_table_{episode}.pkl')

# 保存最终模型
os.makedirs('models', exist_ok=True)
agent.save('models/q_table_final.pkl')

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Episode Reward')
plt.plot(avg_rewards, label='Average Reward (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q-learning Training Curve')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
print("训练曲线已保存到 training_curve.png")

# 关闭环境
env.close()
