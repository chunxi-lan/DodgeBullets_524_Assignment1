import numpy as np
import matplotlib.pyplot as plt
from env import AntiAircraftEnv
from agent import QLearningAgent
import os

# Training parameters
EPISODES = 20000  # Train for 20,000 episodes
MAX_STEPS = 400  # Double steps, provide more shooting opportunities

# Create environment and agent
env = AntiAircraftEnv()
agent = QLearningAgent(state_dim=env.state_dim, action_dim=env.action_dim)

# Record rewards
rewards = []
avg_rewards = []

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    
    for step in range(MAX_STEPS):
        # Select action
        action = agent.act(state)
        
        # Execute action
        next_state, reward, done, _ = env.step(action)
        
        # Learn
        agent.learn(state, action, reward, next_state, done)
        
        # Update state and reward
        state = next_state
        total_reward += reward
        
        # Render every 100 episodes
        if episode % 100 == 0:
            env.render()
    
    # Record reward
    rewards.append(total_reward)
    avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    avg_rewards.append(avg_reward)
    
    # Print training info
    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Save model every 200 episodes
    if episode % 200 == 0:
        os.makedirs('models', exist_ok=True)
        agent.save(f'models/q_table_{episode}.pkl')

# Save final model
os.makedirs('models', exist_ok=True)
agent.save('models/q_table_final.pkl')

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Episode Reward')
plt.plot(avg_rewards, label='Average Reward (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q-learning Training Curve')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
print("Training curve saved to training_curve.png")

# Close environment
env.close()
