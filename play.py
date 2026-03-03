import pygame
import numpy as np
from env import AntiAircraftEnv
from agent import QLearningAgent

# Create environment and agent
env = AntiAircraftEnv()
agent = QLearningAgent(state_dim=env.state_dim, action_dim=env.action_dim)

# Load trained model
try:
    agent.load('models/q_table_final.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found, please run training script first.")
    exit()

# Game main loop
running = True
total_reward = 0
step = 0
fire_count = 0
hit_count = 0

print("Game Controls:")
print("- AI automatically adjusts cannon angle")
print("- Continuous fire mode (all actions include firing)")
print("- Press Q to quit")
print("- Observe how AI adjusts cannon angle and fires continuously")

# Set AI to explore more aggressively (reduce epsilon)
agent.epsilon = 0.05  # Keep some randomness

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
    
    # Get current state
    state = env.get_state()
    
    # AI controls angle adjustment (all actions include firing)
    action = agent.act(state, eval_mode=False)
    
    # Execute action
    next_state, reward, done, _ = env.step(action)
    
    # Count fire times (based on cooldown)
    if env.current_cooldown == env.cooldown:  # Just fired
        fire_count += 1
        print(f"\nAI Fires! Fire count: {fire_count}")
        print(f"Plane position: {env.plane_x:.0f}, Cannon angle: {env.cannon_angle:.1f}°")
    
    # Check if hit
    if reward > 50:  # Hit reward is usually 100
        hit_count += 1
        print(f"*** Hit plane! Total hits: {hit_count} ***")
    
    # Update reward
    total_reward += reward
    step += 1
    
    # Clear screen
    env.screen.fill(env.WHITE)
    
    # Draw plane
    pygame.draw.rect(env.screen, env.BLUE, (env.plane_x, env.plane_y, 50, 25))
    
    # Draw cannon base
    pygame.draw.circle(env.screen, env.BLACK, (env.cannon_x, env.cannon_y), 20)
    
    # Draw cannon barrel
    angle_rad = np.radians(env.cannon_angle)
    cannon_length = 40
    cannon_end_x = env.cannon_x + int(np.sin(angle_rad) * cannon_length)
    cannon_end_y = env.cannon_y - int(np.cos(angle_rad) * cannon_length)
    pygame.draw.line(env.screen, env.BLACK, (env.cannon_x, env.cannon_y), (cannon_end_x, cannon_end_y), 5)
    
    # Draw bullets (using more visible style)
    for bullet in env.bullets:
        # Red solid circle
        pygame.draw.circle(env.screen, (255, 0, 0), (int(bullet.x), int(bullet.y)), bullet.radius)
        # Yellow center dot
        pygame.draw.circle(env.screen, (255, 255, 0), (int(bullet.x), int(bullet.y)), 2)
    
    # Draw firing range
    pygame.draw.circle(env.screen, (200, 200, 200), (env.cannon_x, env.cannon_y), env.fire_range, 1)
    
    # Display info on screen
    font = pygame.font.Font(None, 36)
    
    # First line: step and reward
    text = font.render(f"Step: {step}, Reward: {total_reward:.2f}", True, env.BLACK)
    env.screen.blit(text, (10, 10))
    
    # Second line: control mode
    mode_text = font.render(f"AI Control - Auto Fire", True, env.BLACK)
    env.screen.blit(mode_text, (10, 50))
    
    # Third line: statistics
    stats_text = font.render(f"Bullets: {len(env.bullets)}, Fires: {fire_count}, Hits: {hit_count}", True, env.BLACK)
    env.screen.blit(stats_text, (10, 90))
    
    # Fourth line: status info
    state_text = font.render(f"Plane: {env.plane_x:.0f}, Angle: {env.cannon_angle:.1f}°, Continuous Fire", True, env.BLACK)
    env.screen.blit(state_text, (10, 130))
    
    # Update display
    pygame.display.flip()
    env.clock.tick(60)
    
    # Reset game every 400 steps
    if step >= 400:
        print(f"\n--- Round Ended ---")
        print(f"Total steps: {step}, Total reward: {total_reward:.2f}")
        print(f"Fire count: {fire_count}, Hit count: {hit_count}")
        state = env.reset()
        total_reward = 0
        step = 0
        fire_count = 0
        hit_count = 0

print("\nDemo ended")
env.close()
