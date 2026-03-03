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
        # Initialize pygame
        pygame.init()
        
        # Screen settings
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Dodge Bullets")
        
        # Color definitions
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Game parameters
        self.plane_speed = 3  # Plane speed (consistent with user version)
        self.cannon_angle = 0  # Cannon angle, range -90 to 90 degrees
        self.max_angle = 90  # Expand angle range to 90 degrees
        self.fire_range = 300  # Firing range
        self.hit_angle_threshold = 35  # Hit angle threshold (significantly increased for easier hits)
        self.bullet_speed = 8  # Bullet speed (increased for faster target acquisition)
        self.cooldown = 2  # Fire cooldown (frames) - significantly reduced for higher fire rate
        self.current_cooldown = 0  # Current cooldown
        
        # Plane movement boundaries (local range, consistent with user version)
        self.plane_min_x = 200
        self.plane_max_x = 600
        self.plane_min_y = 50
        self.plane_max_y = 200
        
        # Plane random movement parameters
        self.plane_move_timer = 0
        self.plane_move_direction = [0, 0]  # [dx, dy]
        
        # Bullet list
        self.bullets = []
        
        # Game object positions
        self.cannon_x = self.screen_width // 2
        self.cannon_y = self.screen_height - 50
        self.plane_y = 100
        self.plane_x = np.random.randint(0, self.screen_width)
        
        # State and action space
        self.state_dim = 3  # Normalized plane x-coordinate, normalized cannon angle, plane in range
        self.action_dim = 3  # 0: turn left + fire, 1: turn right + fire, 2: wait + fire
        
        # Other
        self.action = 0
        
    def reset(self):
        # Reset game state (plane initial position consistent with user version)
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
        
        # Process actions (all actions include firing)
        if action == 0:  # Turn left + fire
            self.cannon_angle = max(-self.max_angle, self.cannon_angle - 5)
        elif action == 1:  # Turn right + fire
            self.cannon_angle = min(self.max_angle, self.cannon_angle + 5)
        elif action == 2:  # Wait + fire
            pass  # Only fire, no rotation
        
        # Continuous firing (all actions fire)
        if self.current_cooldown == 0:
            # Calculate cannon muzzle position
            angle_rad = np.radians(self.cannon_angle)
            cannon_length = 40
            cannon_end_x = self.cannon_x + int(np.sin(angle_rad) * cannon_length)
            cannon_end_y = self.cannon_y - int(np.cos(angle_rad) * cannon_length)
            # Create bullet
            bullet = Bullet(cannon_end_x, cannon_end_y, self.cannon_angle, self.bullet_speed)
            self.bullets.append(bullet)
            self.current_cooldown = self.cooldown
        
        # Update cooldown
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
        
        # Plane random movement (simulate user behavior, no movement for first 0.2 seconds)
        self.plane_move_timer += 1
        
        # No movement for first 0.2 seconds (12 frames)
        if self.plane_move_timer <= 12:
            self.plane_move_direction = [0, 0]
        # Random movement after that
        elif self.plane_move_timer >= 30:
            self.plane_move_timer = 13  # Reset timer but keep above 13 to avoid re-entering first 12 frames
            # Randomly choose movement direction (increase stop probability)
            # 0: stop, -1: left/up, 1: right/down
            # 40% probability to stop
            if np.random.random() < 0.4:
                # Stop
                self.plane_move_direction = [0, 0]
            else:
                # Random movement
                self.plane_move_direction = [
                    np.random.choice([-1, 1]),  # x direction (excluding 0 to ensure movement)
                    np.random.choice([-1, 1])   # y direction (excluding 0 to ensure movement)
                ]
        
        # Move plane (limit to local range)
        new_x = self.plane_x + self.plane_move_direction[0] * self.plane_speed
        new_y = self.plane_y + self.plane_move_direction[1] * self.plane_speed
        
        # Boundary check
        self.plane_x = max(self.plane_min_x, min(self.plane_max_x, new_x))
        self.plane_y = max(self.plane_min_y, min(self.plane_max_y, new_y))
        
        # Update bullet positions
        hit = False
        for bullet in self.bullets[:]:
            bullet.update()
            
            # Detect bullet-plane collision (only when check_collision is True)
            if check_collision and self._check_bullet_hit(bullet):
                hit = True
                bullet.active = False
                self.bullets.remove(bullet)
            
            # Remove bullets out of screen
            elif bullet.is_out_of_bounds(self.screen_width, self.screen_height):
                bullet.active = False
                self.bullets.remove(bullet)
        
        # Calculate reward
        reward = self._calculate_reward(hit)
        
        # Check if done (end episode every 100 steps)
        done = False
        
        return self.get_state(), reward, done, {}
    
    def render(self):
        # Draw background
        self.screen.fill(self.WHITE)
        
        # Draw plane
        pygame.draw.rect(self.screen, self.RED, (self.plane_x, self.plane_y, 50, 25))
        
        # Draw cannon base
        pygame.draw.circle(self.screen, self.BLACK, (self.cannon_x, self.cannon_y), 20)
        
        # Draw cannon barrel
        angle_rad = np.radians(self.cannon_angle)
        cannon_length = 40
        cannon_end_x = self.cannon_x + int(np.sin(angle_rad) * cannon_length)
        cannon_end_y = self.cannon_y - int(np.cos(angle_rad) * cannon_length)
        pygame.draw.line(self.screen, self.BLACK, (self.cannon_x, self.cannon_y), (cannon_end_x, cannon_end_y), 5)
        
        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(self.screen)
        
        # Draw firing range
        pygame.draw.circle(self.screen, (200, 200, 200), (self.cannon_x, self.cannon_y), self.fire_range, 1)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
    
    def get_state(self):
        # Normalize state
        plane_x_norm = self.plane_x / self.screen_width
        angle_norm = self.cannon_angle / self.max_angle
        in_range = 1.0 if abs(self.plane_x - self.cannon_x) < self.fire_range else 0.0
        return np.array([plane_x_norm, angle_norm, in_range], dtype=np.float32)
    
    def _check_bullet_hit(self, bullet):
        # Check if bullet hits plane (with tolerance)
        plane_left = self.plane_x - bullet.radius
        plane_right = self.plane_x + 50 + bullet.radius
        plane_top = self.plane_y - bullet.radius
        plane_bottom = self.plane_y + 25 + bullet.radius
        
        return (bullet.x >= plane_left and bullet.x <= plane_right and
                bullet.y >= plane_top and bullet.y <= plane_bottom)
    
    def _check_hit(self):
        # Check if hit (kept for compatibility)
        if abs(self.plane_x - self.cannon_x) > self.fire_range:
            return False
        
        # Calculate angle difference
        target_angle = np.degrees(np.arctan2(self.plane_x - self.cannon_x, self.cannon_y - self.plane_y))
        angle_diff = abs(target_angle - self.cannon_angle)
        
        return angle_diff < self.hit_angle_threshold
    
    def _calculate_reward(self, hit):
        reward = -0.005  # Per-step survival penalty (very low)
        if hit:
            reward += 100.0  # Significantly increased hit reward
        
        # Calculate target angle based on plane position
        dx = self.plane_x - self.cannon_x
        dy = self.cannon_y - self.plane_y
        target_angle = np.degrees(np.arctan2(dx, dy))
        target_angle = max(-self.max_angle, min(self.max_angle, target_angle))
        
        # Reward for aiming at plane (based on angle difference)
        angle_diff = abs(self.cannon_angle - target_angle)
        if angle_diff < 5:  # Angle difference less than 5 degrees
            reward += 2.0  # Significantly increased aiming reward
        elif angle_diff < 10:  # Angle difference less than 10 degrees
            reward += 1.0
        elif angle_diff < 20:  # Angle difference less than 20 degrees
            reward += 0.5
        
        # Encourage correct turning direction
        if self.action == 0:  # Turn left + fire
            if target_angle < self.cannon_angle:  # Need to turn left
                reward += 0.3  # Reward correct turning
        elif self.action == 1:  # Turn right + fire
            if target_angle > self.cannon_angle:  # Need to turn right
                reward += 0.3  # Reward correct turning
        
        return reward
    
    def close(self):
        pygame.quit()
