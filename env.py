import pygame
import numpy as np

class Bullet:
    def __init__(self, x, y, angle, speed=10):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.radius = 5
        self.active = True
        
    def update(self):
        angle_rad = np.radians(self.angle)
        self.x += np.sin(angle_rad) * self.speed
        self.y -= np.cos(angle_rad) * self.speed
        
    def is_out_of_bounds(self, screen_width, screen_height):
        return (self.x < 0 or self.x > screen_width or 
                self.y < 0 or self.y > screen_height)
    
    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.radius)

class AntiAircraftEnv:
    def __init__(self):
        # 初始化pygame
        pygame.init()
        
        # 屏幕设置
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("防空炮打飞机")
        
        # 颜色定义
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # 时钟
        self.clock = pygame.time.Clock()
        
        # 游戏参数
        self.plane_speed = 3  # 飞机速度（与用户版本一致）
        self.cannon_angle = 0  # 炮口角度，范围-90到90度
        self.max_angle = 90  # 扩大角度范围到90度
        self.fire_range = 300  # 射程
        self.hit_angle_threshold = 35  # 命中角度阈值（大幅增加，更容易命中）
        self.bullet_speed = 8  # 子弹速度（增加速度，更快到达目标）
        self.cooldown = 2  # 开火冷却时间（帧数）- 大幅减少冷却时间，提高射击频率
        self.current_cooldown = 0  # 当前冷却时间
        
        # 飞机移动边界（局部范围，与用户版本一致）
        self.plane_min_x = 200
        self.plane_max_x = 600
        self.plane_min_y = 50
        self.plane_max_y = 200
        
        # 飞机随机移动参数
        self.plane_move_timer = 0
        self.plane_move_direction = [0, 0]  # [dx, dy]
        
        # 子弹列表
        self.bullets = []
        
        # 游戏对象位置
        self.cannon_x = self.screen_width // 2
        self.cannon_y = self.screen_height - 50
        self.plane_y = 100
        self.plane_x = np.random.randint(0, self.screen_width)
        
        # 状态和动作空间
        self.state_dim = 3  # 飞机x坐标归一化、炮口角度归一化、飞机是否在射程内
        self.action_dim = 3  # 0: 左转+开火, 1: 右转+开火, 2: 等待+开火
        
        # 其他
        self.action = 0
        
    def reset(self):
        # 重置游戏状态（飞机初始位置与用户版本一致）
        self.plane_x = self.screen_width // 2
        self.plane_y = 100
        self.cannon_angle = 0
        self.action = 0
        self.bullets = []
        self.current_cooldown = 0
        self.plane_move_timer = 0
        self.plane_move_direction = [0, 0]
        return self.get_state()
    
    def step(self, action, check_collision=True):
        self.action = action
        
        # 处理动作（所有动作都包含开火）
        if action == 0:  # 左转+开火
            self.cannon_angle = max(-self.max_angle, self.cannon_angle - 5)
        elif action == 1:  # 右转+开火
            self.cannon_angle = min(self.max_angle, self.cannon_angle + 5)
        elif action == 2:  # 等待+开火
            pass  # 只开火，不转动
        
        # 持续开火（所有动作都开火）
        if self.current_cooldown == 0:
            # 计算炮口位置
            angle_rad = np.radians(self.cannon_angle)
            cannon_length = 40
            cannon_end_x = self.cannon_x + int(np.sin(angle_rad) * cannon_length)
            cannon_end_y = self.cannon_y - int(np.cos(angle_rad) * cannon_length)
            # 创建子弹
            bullet = Bullet(cannon_end_x, cannon_end_y, self.cannon_angle, self.bullet_speed)
            self.bullets.append(bullet)
            self.current_cooldown = self.cooldown
        
        # 更新冷却时间
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
        
        # 飞机随机移动（模拟用户行为，前0.2秒不移动）
        self.plane_move_timer += 1
        
        # 前0.2秒（12帧）不移动
        if self.plane_move_timer <= 12:
            self.plane_move_direction = [0, 0]
        # 之后随机移动
        elif self.plane_move_timer >= 30:
            self.plane_move_timer = 13  # 重置计时器，但保持在13以上，避免再次进入前12帧
            # 随机选择移动方向（增加停止的概率）
            # 0: 停止, -1: 左/上, 1: 右/下
            # 停止的概率为40%
            if np.random.random() < 0.4:
                # 停止
                self.plane_move_direction = [0, 0]
            else:
                # 随机移动
                self.plane_move_direction = [
                    np.random.choice([-1, 1]),  # x方向（不包括0，确保移动）
                    np.random.choice([-1, 1])   # y方向（不包括0，确保移动）
                ]
        
        # 移动飞机（限制在局部范围内）
        new_x = self.plane_x + self.plane_move_direction[0] * self.plane_speed
        new_y = self.plane_y + self.plane_move_direction[1] * self.plane_speed
        
        # 边界检查
        self.plane_x = max(self.plane_min_x, min(self.plane_max_x, new_x))
        self.plane_y = max(self.plane_min_y, min(self.plane_max_y, new_y))
        
        # 更新子弹位置
        hit = False
        for bullet in self.bullets[:]:
            bullet.update()
            
            # 检测子弹与飞机的碰撞（仅在check_collision为True时检测）
            if check_collision and self._check_bullet_hit(bullet):
                hit = True
                bullet.active = False
                self.bullets.remove(bullet)
            
            # 移除飞出屏幕的子弹
            elif bullet.is_out_of_bounds(self.screen_width, self.screen_height):
                bullet.active = False
                self.bullets.remove(bullet)
        
        # 计算奖励
        reward = self._calculate_reward(hit)
        
        # 检查是否结束（每100步结束一个回合）
        done = False
        
        return self.get_state(), reward, done, {}
    
    def render(self):
        # 绘制背景
        self.screen.fill(self.WHITE)
        
        # 绘制飞机
        pygame.draw.rect(self.screen, self.RED, (self.plane_x, self.plane_y, 50, 25))
        
        # 绘制炮台
        pygame.draw.circle(self.screen, self.BLACK, (self.cannon_x, self.cannon_y), 20)
        
        # 绘制炮管
        angle_rad = np.radians(self.cannon_angle)
        cannon_length = 40
        cannon_end_x = self.cannon_x + int(np.sin(angle_rad) * cannon_length)
        cannon_end_y = self.cannon_y - int(np.cos(angle_rad) * cannon_length)
        pygame.draw.line(self.screen, self.BLACK, (self.cannon_x, self.cannon_y), (cannon_end_x, cannon_end_y), 5)
        
        # 绘制子弹
        for bullet in self.bullets:
            bullet.draw(self.screen)
        
        # 绘制射程范围
        pygame.draw.circle(self.screen, (200, 200, 200), (self.cannon_x, self.cannon_y), self.fire_range, 1)
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(60)
    
    def get_state(self):
        # 归一化状态
        plane_x_norm = self.plane_x / self.screen_width
        angle_norm = self.cannon_angle / self.max_angle
        in_range = 1.0 if abs(self.plane_x - self.cannon_x) < self.fire_range else 0.0
        return np.array([plane_x_norm, angle_norm, in_range], dtype=np.float32)
    
    def _check_bullet_hit(self, bullet):
        # 检查子弹是否击中飞机（增加容差范围）
        plane_left = self.plane_x - bullet.radius
        plane_right = self.plane_x + 50 + bullet.radius
        plane_top = self.plane_y - bullet.radius
        plane_bottom = self.plane_y + 25 + bullet.radius
        
        return (bullet.x >= plane_left and bullet.x <= plane_right and
                bullet.y >= plane_top and bullet.y <= plane_bottom)
    
    def _check_hit(self):
        # 检查是否命中（保留用于兼容性）
        if abs(self.plane_x - self.cannon_x) > self.fire_range:
            return False
        
        # 计算角度差
        target_angle = np.degrees(np.arctan2(self.plane_x - self.cannon_x, self.cannon_y - self.plane_y))
        angle_diff = abs(target_angle - self.cannon_angle)
        
        return angle_diff < self.hit_angle_threshold
    
    def _calculate_reward(self, hit):
        reward = -0.005  # 每步生存惩罚（极低）
        if hit:
            reward += 100.0  # 击中奖励极大增加
        
        # 计算炮塔应该指向的角度（基于飞机位置）
        dx = self.plane_x - self.cannon_x
        dy = self.cannon_y - self.plane_y
        target_angle = np.degrees(np.arctan2(dx, dy))
        target_angle = max(-self.max_angle, min(self.max_angle, target_angle))
        
        # 炮口对准飞机奖励（基于角度差异）
        angle_diff = abs(self.cannon_angle - target_angle)
        if angle_diff < 5:  # 角度差异小于5度
            reward += 2.0  # 极大增加瞄准奖励
        elif angle_diff < 10:  # 角度差异小于10度
            reward += 1.0
        elif angle_diff < 20:  # 角度差异小于20度
            reward += 0.5
        
        # 鼓励向正确方向转动
        if self.action == 0:  # 左转+开火
            if target_angle < self.cannon_angle:  # 需要左转
                reward += 0.3  # 奖励正确的转向
        elif self.action == 1:  # 右转+开火
            if target_angle > self.cannon_angle:  # 需要右转
                reward += 0.3  # 奖励正确的转向
        
        return reward
    
    def close(self):
        pygame.quit()
