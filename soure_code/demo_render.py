import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
from enum import Enum
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

BLOCK_SIZE = 20
SPEED = 1000

WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, w=640, h=480, render_mode=True):
        super(SnakeEnv, self).__init__()
        self.w = w
        self.h = h
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)
        if self.render_mode:
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake PPO Demo')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.env_snake_dir = Direction.LEFT
        self.env_snake_head = Point(self.w // 3, self.h // 3)
        self.env_snake = [
            self.env_snake_head,
            Point(self.env_snake_head.x + BLOCK_SIZE, self.env_snake_head.y)
        ]
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        self._place_bomb()
        return self._get_observation(), {}

    def step(self, action):
        self.frame_iteration += 1
        reward = 0.0
        terminated = False
        truncated = False
        self._move(action)
        self.snake.insert(0, self.head)
        self._move_env_snake()
        self.env_snake.insert(0, self.env_snake_head)
        if self._is_collision(self.head):
            terminated = True
            reward = -10.0
        if not terminated and self.frame_iteration > 100 * len(self.snake):
            truncated = True
            reward = -10.0
        if not terminated and not truncated:
            if self.head == self.food:
                self.score += 1
                reward = 10.0
                self._place_food()
            else:
                self.snake.pop()
        if not terminated and not truncated and self.head == self.bomb:
            terminated = True
            reward = -10.0
        if not terminated and not truncated:
            if self._env_snake_collision(self.env_snake_head):
                self._respawn_env_snake()
            else:
                self.env_snake.pop()
        if self.render_mode:
            self._update_ui()
        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, {}

    def render(self, mode="rgb_array"):
        if not self.render_mode:
            return None
        frame = pygame.surfarray.array3d(self.display)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

    def _place_bomb(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.bomb = Point(x, y)

    def _move_env_snake(self):
        action = random.randint(0, 2)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.env_snake_dir)
        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.env_snake_dir = new_dir
        x = self.env_snake_head.x
        y = self.env_snake_head.y
        if self.env_snake_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.env_snake_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.env_snake_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.env_snake_dir == Direction.UP:
            y -= BLOCK_SIZE
        self.env_snake_head = Point(x, y)

    def _env_snake_collision(self, pt):
        if (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or
            pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True
        if pt == self.bomb:
            return True
        return False

    def _respawn_env_snake(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.env_snake_head = Point(x, y)
        self.env_snake_dir = random.choice(list(Direction))
        self.env_snake = [self.env_snake_head]

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        if pt in self.env_snake:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, (pt.x+4, pt.y+4, 12, 12))
        for pt in self.env_snake:
            pygame.draw.rect(self.display, (255, 165, 0), (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (255, 215, 0), (pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, GREEN, (self.bomb.x, self.bomb.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, (self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(SPEED)

    def _get_observation(self):
        dx_food = (self.food.x - self.head.x) / float(self.w)
        dy_food = (self.food.y - self.head.y) / float(self.h)
        dx_bomb = (self.bomb.x - self.head.x) / float(self.w)
        dy_bomb = (self.bomb.y - self.head.y) / float(self.h)
        dir_r = 1 if self.direction == Direction.RIGHT else 0
        dir_l = 1 if self.direction == Direction.LEFT else 0
        dir_u = 1 if self.direction == Direction.UP else 0
        dir_d = 1 if self.direction == Direction.DOWN else 0
        env_dx = (self.env_snake_head.x - self.head.x) / float(self.w)
        env_dy = (self.env_snake_head.y - self.head.y) / float(self.h)
        danger_straight = 1 if self._will_collide_if_straight() else 0
        snake_len = len(self.snake) / 100.0
        state = np.array([
            dx_food, dy_food,
            dx_bomb, dy_bomb,
            dir_r, dir_l, dir_u, dir_d,
            env_dx, env_dy,
            danger_straight,
            snake_len,
            0.0, 0.0, 0.0
        ], dtype=np.float32)
        return state

    def _will_collide_if_straight(self):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        new_dir = clock_wise[idx]
        x = self.head.x
        y = self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
        pt = Point(x, y)
        return self._is_collision(pt)
    
    
    
    
# ORIGINAL VER => realtime rendering
# 
# if __name__ == "__main__":
#     # Demonstration phase: use render_mode=True so the game window shows.
#     demo_env = SnakeEnv(render_mode=True)
#     obs, info = demo_env.reset()
#     # Wrap in DummyVecEnv if needed (optional for demonstration)
#     # Here we just use the environment directly.
#     model = PPO.load("./ppo_snake_final_3.zip")  # Load the model saved during training

#      # Test the trained model with rendering
#     test_env = SnakeEnv(render_mode=True)
#     obs, info = test_env.reset()
    
    
#     total_episodes = 24  # Change this to the number of episodes you want to run
#     for episode in range(total_episodes):
#         total_reward = 0
#         done = False
#         truncated = False
#         obs, info = test_env.reset()
#         while not (done or truncated):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, truncated, info = test_env.step(action)
#             total_reward += reward
#             test_env.render()  # Updates the display
#         print(f"Episode {episode+1} finished! Total reward: {total_reward}")




def demo_best_episode(model_path="./ppo_snake_final_3.zip", n_episodes=50):
    demo_env = SnakeEnv(render_mode=True)
    
    # Load the trained model:
    #   REMINDER => remember to change the model paths since we running
    #               the models one by one cuz we dont want models overlapping in the log file and
    #               also the finla zip file witht final model. 
    model = PPO.load(model_path)
    
    best_reward = -float('inf')
    best_frames = None
    best_episode = 0
    
    for episode in range(1, n_episodes+1):
        frames = []
        obs, info = demo_env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        # recoring while runnning epidsodes
        while not (done or truncated):
            frame = demo_env.render(mode="rgb_array")
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = demo_env.step(action)
            total_reward += reward
            
        print(f"Episode {episode} finished! Total reward: {total_reward}")
        
        # if this episode has the best reward so far => store its frames.
        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
            best_episode = episode

    print(f"Best episode was {best_episode} with a reward of {best_reward}")
    
    replay_best_episode(best_frames)

def replay_best_episode(frames, delay=30):
    pygame.init()
    # Get the size from the first frame.
    height, width, _ = frames[0].shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Best Episode Replay")
    
    clock = pygame.time.Clock()
    running = True
    frame_index = 0
    n_frames = len(frames)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # go through teheframes once
        if frame_index < n_frames:
            #NOTE:
            #      Pygame expects the array in (width, height, channels)
            frame_surface = pygame.surfarray.make_surface(np.transpose(frames[frame_index], (1, 0, 2)))
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            frame_index += 1
            clock.tick(1000/delay)
        else:
            # end of replay -> wait until window is closed.
            pygame.time.delay(2000)
            running = False
    pygame.quit()

if __name__ == "__main__":
    demo_best_episode(model_path="./ppo_snake_final_2.zip", n_episodes=30)