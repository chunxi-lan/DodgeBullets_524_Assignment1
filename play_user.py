import pygame
import numpy as np
from env import AntiAircraftEnv
from agent import QLearningAgent
import os

# Initialize Pygame
pygame.init()

# Game parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
PLAYER_PLANE_SPEED = 3
PLAYER_HEALTH = 100  # Health points, each hit costs 1 HP
GAME_DURATION = 1200  # 20 seconds (60 FPS * 20 seconds)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

class PlayerPlane:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 50
        self.height = 25
        self.speed = PLAYER_PLANE_SPEED
        self.health = PLAYER_HEALTH
        self.max_health = PLAYER_HEALTH
        
        # Movement boundaries (restricted area)
        self.min_x = 200
        self.max_x = SCREEN_WIDTH - 200
        self.min_y = 50
        self.max_y = 200
    
    def move(self, keys):
        # WASD controls (up, down, left, right)
        if keys[pygame.K_w]:  # W - Move up
            self.y = max(self.min_y, self.y - self.speed)
        if keys[pygame.K_s]:  # S - Move down
            self.y = min(self.max_y, self.y + self.speed)
        if keys[pygame.K_a]:  # A - Move left
            self.x = max(self.min_x, self.x - self.speed)
        if keys[pygame.K_d]:  # D - Move right
            self.x = min(self.max_x, self.x + self.speed)
    
    def take_damage(self, damage):
        self.health = max(0, self.health - damage)
    
    def is_alive(self):
        return self.health > 0
    
    def draw(self, screen):
        # Draw player plane
        pygame.draw.rect(screen, GREEN, (self.x, self.y, self.width, self.height))
        
        # Draw health bar background
        bar_width = 60
        bar_height = 8
        bar_x = self.x + (self.width - bar_width) // 2
        bar_y = self.y - 15
        
        pygame.draw.rect(screen, GRAY, (bar_x, bar_y, bar_width, bar_height))
        
        # Draw health bar
        health_ratio = self.health / self.max_health
        health_bar_width = int(bar_width * health_ratio)
        health_color = GREEN if health_ratio > 0.5 else (255, 255, 0) if health_ratio > 0.25 else RED
        pygame.draw.rect(screen, health_color, (bar_x, bar_y, health_bar_width, bar_height))

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dodge Bullets")
clock = pygame.time.Clock()

# Create AI environment and agent
env = AntiAircraftEnv()
agent = QLearningAgent(state_dim=env.state_dim, action_dim=env.action_dim)

# Load trained model
model_path = 'models/q_table_final.pkl'
if os.path.exists(model_path):
    agent.load(model_path)
    print("Model loaded successfully!")
else:
    print("Model not found, using random strategy")

# Create player plane
player_plane = PlayerPlane(SCREEN_WIDTH // 2, 100)

# Game state
game_frame = 0
game_over = False
victory = False
game_started = False  # Whether game has started

# Game main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_SPACE and not game_started:
                game_started = True
                print("Game Started!")
    
    # Display start screen
    if not game_started:
        screen.fill(WHITE)
        font = pygame.font.Font(None, 48)
        
        # Title
        title_text = font.render("Dodge Bullets", True, BLACK)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 150))
        screen.blit(title_text, title_rect)
        
        # Instructions
        small_font = pygame.font.Font(None, 36)
        instructions = [
            "Instructions:",
            "",
            "W - Move Up",
            "A - Move Left", 
            "S - Move Down",
            "D - Move Right",
            "",
            "Survive for 20 seconds to win!",
            "You have 100 health points.",
            "Each hit costs 1 HP.",
            "",
            "Press SPACE to start",
            "Press Q to quit"
        ]
        
        y_offset = 220
        for line in instructions:
            if line == "Instructions:":
                text = small_font.render(line, True, RED)
            elif line == "Press SPACE to start":
                text = small_font.render(line, True, GREEN)
            else:
                text = small_font.render(line, True, BLACK)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            screen.blit(text, text_rect)
            y_offset += 30
        
        pygame.display.flip()
        clock.tick(FPS)
        continue
    
    if not game_over:
        # Get key presses
        keys = pygame.key.get_pressed()
        
        # Player controls plane
        player_plane.move(keys)
        
        # Update plane position in environment
        env.plane_x = player_plane.x
        env.plane_y = player_plane.y
        
        # AI controls cannon
        state = env.get_state()
        action = agent.act(state, eval_mode=True)
        
        # Execute action (continuous fire, disable collision detection in env)
        next_state, reward, done, _ = env.step(action, check_collision=False)
        
        # Check if bullets hit player plane
        # Note: Bullet positions are already updated in env.step(), only need collision detection here
        for bullet in env.bullets[:]:
            # Check collision (bullet hit volume matches plane size)
            # Bullet is treated as a point, check if it's within plane rectangle
            if (bullet.x >= player_plane.x and 
                bullet.x <= player_plane.x + player_plane.width and
                bullet.y >= player_plane.y and 
                bullet.y <= player_plane.y + player_plane.height):
                player_plane.take_damage(1)  # Each hit costs 1 HP
                bullet.active = False
                env.bullets.remove(bullet)
                print(f"Hit! Health: {player_plane.health}")
            
            # Remove bullets that fly off screen
            elif bullet.is_out_of_bounds(env.screen_width, env.screen_height):
                bullet.active = False
                env.bullets.remove(bullet)
        
        # Update game frame
        game_frame += 1
        
        # Check victory condition
        if game_frame >= GAME_DURATION:
            victory = True
            game_over = True
            print("\n=== VICTORY! You survived for 20 seconds! ===")
        
        # Check defeat condition
        if not player_plane.is_alive():
            game_over = True
            print("\n=== GAME OVER! Plane destroyed ===")
    
    # Clear screen
    screen.fill(WHITE)
    
    # Draw AI environment (cannon, bullets, etc.)
    # Note: Don't draw env.plane (blue plane) as player has their own green plane
    pygame.draw.circle(screen, BLACK, (env.cannon_x, env.cannon_y), 20)
    
    angle_rad = np.radians(env.cannon_angle)
    cannon_length = 40
    cannon_end_x = env.cannon_x + int(np.sin(angle_rad) * cannon_length)
    cannon_end_y = env.cannon_y - int(np.cos(angle_rad) * cannon_length)
    pygame.draw.line(screen, BLACK, (env.cannon_x, env.cannon_y), (cannon_end_x, cannon_end_y), 5)
    
    for bullet in env.bullets:
        bullet.draw(screen)
    
    # Draw player plane
    player_plane.draw(screen)
    
    # Draw movement boundary
    pygame.draw.rect(screen, GRAY, (player_plane.min_x, player_plane.min_y, 
                                      player_plane.max_x - player_plane.min_x + player_plane.width, 
                                      player_plane.max_y - player_plane.min_y + player_plane.height), 1)
    
    # Draw game status
    font = pygame.font.Font(None, 36)
    
    # Remaining time
    time_left = max(0, (GAME_DURATION - game_frame) // FPS)
    time_text = font.render(f"Time: {time_left}s", True, BLACK)
    screen.blit(time_text, (10, 10))
    
    # Health
    health_text = font.render(f"Health: {player_plane.health}/{player_plane.max_health}", True, BLACK)
    screen.blit(health_text, (10, 50))
    
    # Game over message
    if game_over:
        game_over_font = pygame.font.Font(None, 72)
        if victory:
            result_text = game_over_font.render("VICTORY!", True, GREEN)
        else:
            result_text = game_over_font.render("GAME OVER", True, RED)
        text_rect = result_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(result_text, text_rect)
    
    # Update display
    pygame.display.flip()
    clock.tick(FPS)

print("\nGame ended")
pygame.quit()
