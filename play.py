import pygame
import numpy as np
from env import AntiAircraftEnv
from agent import QLearningAgent

# 创建环境和智能体
env = AntiAircraftEnv()
agent = QLearningAgent(state_dim=env.state_dim, action_dim=env.action_dim)

# 加载训练好的模型
try:
    agent.load('models/q_table_final.pkl')
    print("模型加载成功！")
except FileNotFoundError:
    print("模型文件未找到，请先运行训练脚本。")
    exit()

# 游戏主循环
running = True
total_reward = 0
step = 0
fire_count = 0
hit_count = 0

print("游戏控制：")
print("- AI自动调整炮塔角度")
print("- 持续开火模式（所有动作都包含开火）")
print("- 按Q键退出")
print("- 观察AI如何调整炮口角度并持续开火")

# 设置AI更积极地探索（降低epsilon）
agent.epsilon = 0.05  # 保留一点随机性

while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
    
    # 获取当前状态
    state = env.get_state()
    
    # AI控制角度调整（所有动作都包含开火）
    action = agent.act(state, eval_mode=False)
    
    # 执行动作
    next_state, reward, done, _ = env.step(action)
    
    # 统计开火次数（基于冷却时间）
    if env.current_cooldown == env.cooldown:  # 刚刚开火
        fire_count += 1
        print(f"\nAI开火！第{fire_count}次开火")
        print(f"飞机位置: {env.plane_x:.0f}, 炮口角度: {env.cannon_angle:.1f}°")
    
    # 检测是否击中
    if reward > 50:  # 击中奖励通常是100
        hit_count += 1
        print(f"*** 击中飞机！累计击中: {hit_count}次 ***")
    
    # 更新奖励
    total_reward += reward
    step += 1
    
    # 清屏
    env.screen.fill(env.WHITE)
    
    # 绘制飞机
    pygame.draw.rect(env.screen, env.BLUE, (env.plane_x, env.plane_y, 50, 25))
    
    # 绘制炮台
    pygame.draw.circle(env.screen, env.BLACK, (env.cannon_x, env.cannon_y), 20)
    
    # 绘制炮管
    angle_rad = np.radians(env.cannon_angle)
    cannon_length = 40
    cannon_end_x = env.cannon_x + int(np.sin(angle_rad) * cannon_length)
    cannon_end_y = env.cannon_y - int(np.cos(angle_rad) * cannon_length)
    pygame.draw.line(env.screen, env.BLACK, (env.cannon_x, env.cannon_y), (cannon_end_x, cannon_end_y), 5)
    
    # 绘制子弹（使用更显眼的样式）
    for bullet in env.bullets:
        # 红色实心圆
        pygame.draw.circle(env.screen, (255, 0, 0), (int(bullet.x), int(bullet.y)), bullet.radius)
        # 黄色中心点
        pygame.draw.circle(env.screen, (255, 255, 0), (int(bullet.x), int(bullet.y)), 2)
    
    # 绘制射程范围
    pygame.draw.circle(env.screen, (200, 200, 200), (env.cannon_x, env.cannon_y), env.fire_range, 1)
    
    # 在屏幕上显示信息
    font = pygame.font.Font(None, 36)
    
    # 第一行：步数和奖励
    text = font.render(f"Step: {step}, Reward: {total_reward:.2f}", True, env.BLACK)
    env.screen.blit(text, (10, 10))
    
    # 第二行：控制模式
    mode_text = font.render(f"AI Control - Auto Fire", True, env.BLACK)
    env.screen.blit(mode_text, (10, 50))
    
    # 第三行：统计信息
    stats_text = font.render(f"Bullets: {len(env.bullets)}, Fires: {fire_count}, Hits: {hit_count}", True, env.BLACK)
    env.screen.blit(stats_text, (10, 90))
    
    # 第四行：状态信息
    state_text = font.render(f"Plane: {env.plane_x:.0f}, Angle: {env.cannon_angle:.1f}°, Continuous Fire", True, env.BLACK)
    env.screen.blit(state_text, (10, 130))
    
    # 更新显示
    pygame.display.flip()
    env.clock.tick(60)
    
    # 每400步重置游戏
    if step >= 400:
        print(f"\n--- 回合结束 ---")
        print(f"总步数: {step}, 总奖励: {total_reward:.2f}")
        print(f"开火次数: {fire_count}, 击中次数: {hit_count}")
        state = env.reset()
        total_reward = 0
        step = 0
        fire_count = 0
        hit_count = 0

print("\n演示结束")
env.close()
