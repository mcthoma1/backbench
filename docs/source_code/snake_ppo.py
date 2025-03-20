import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
from enum import Enum
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch
import os
import cv2
from datetime import datetime
import math

def log_score(message, log_file="ppo_ultimate_final.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {message}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)

# Settings and Hyperparameters
TIME_COMSUME   = 0.01
TIME_STEP      = 3000000
LEARNING       = 0.0003
CLIP           = 0.2
ENT_COEF       = 0.03
BLOCK_SIZE     = 20
SPEED          = float('inf')  
RECORD_SPEED   = 10000          

# Colors
WHITE  = (255, 255, 255)  
RED    = (200, 0, 0)      
BLUE1  = (0, 0, 255)     
BLUE2  = (0, 100, 255)   
BLACK  = (0, 0, 0)
GREEN  = (0, 255, 0)     
YELLOW = (255, 255, 0)   

# Directions
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

# Custom Snake Environment
class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, w=640, h=480, render_mode=False):
        super(SnakeEnv, self).__init__()
        self.w = w
        self.h = h
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)

        pygame.init()
        self.font = pygame.font.SysFont('arial', 25)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake PPO')
        self.clock = pygame.time.Clock()

        # Video folder
        os.makedirs("ultimate_final_snake_videos", exist_ok=True)

        self.num_bombs = 1

        self.best_score = float('-inf')
        self.best_video_filename = None

        self.episode_count = 0

        self.temp_frames = []

        self.episode_scores = []
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        if hasattr(self, "score"):
            self.episode_scores.append(self.score)
            log_score(f"Episode ended with score: {self.score}, total bombs: {self.num_bombs}")

           
            if (self.score > self.best_score):
                saved_filename = self._save_video(self.temp_frames, self.score)
                
                if self.score > self.best_score:
                    self.best_score = self.score
                    self.best_video_filename = saved_filename

            self.temp_frames = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.episode_count += 1

        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.frame_iteration = 0

        self._place_food()
        self._place_bomb()

        self._init_hardcoded_snake()

        return self._get_observation(), {}

    def step(self, action):
        self.frame_iteration += 1
        reward = 0.0
        terminated = False
        truncated = False

        # ----- Main Snake Movement -----
        self._move(action)
        self.snake.insert(0, self.head)

        if self._is_collision(self.head):
            terminated = True
            reward = -10

        reward -= TIME_COMSUME
        if not terminated and self.frame_iteration > 100 * len(self.snake):
            truncated = True
            reward = -8

        # ----- Main Snake Food Consumption -----
        if not terminated and not truncated:
            if self.head == self.food:
                self.score += 1
                if self.num_bombs < 25:
                    self.num_bombs += 1
                    self._add_bomb()
                reward = 10 * len(self.snake)
                self._place_food()
            else:
                self.snake.pop()

        # ----- Main Snake Bomb Collision -----
        if not terminated and not truncated:
            for bomb in self.bombs:
                if self.head == bomb:
                    terminated = True
                    reward = -10
                    break

        # ----- Secondary Snake Movement -----
        self._move_hardcoded_snake()

        self._update_ui(SPEED)

        frame = self.render()
        self.temp_frames.append(frame)

        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, {}

    def render(self, mode="rgb_array"):
        frame = pygame.surfarray.array3d(self.display)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    # ----- Main Snake Movement Helper -----
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

    # ----- Food and Bomb Placement -----
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        while (self.food in self.snake or 
               (hasattr(self, 'bombs') and self.food in self.bombs) or 
               (hasattr(self, 'snake2') and self.food in self.snake2)):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)

    def _place_bomb(self):
        self.bombs = []
        for _ in range(self.num_bombs):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                bomb = Point(x, y)
                if bomb not in self.snake and bomb != self.food and (not hasattr(self, 'snake2') or bomb not in self.snake2):
                    self.bombs.append(bomb)
                    break

    def _add_bomb(self):
        if self.num_bombs > len(self.bombs):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                bomb = Point(x, y)
                if (bomb not in self.snake and bomb != self.food and 
                    (not hasattr(self, 'snake2') or bomb not in self.snake2) and 
                    bomb not in self.bombs):
                    self.bombs.append(bomb)
                    break

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    # ----- Secondary Snake (Hardcoded Greedy Snake) Methods -----
    def _init_hardcoded_snake(self):
        self.head2 = Point(BLOCK_SIZE * 5, BLOCK_SIZE * 5)
        self.snake2 = [
            self.head2,
            Point(self.head2.x - BLOCK_SIZE, self.head2.y),
            Point(self.head2.x - 2 * BLOCK_SIZE, self.head2.y)
        ]
        self.direction2 = Direction.RIGHT

    def _is_collision_snake2(self, pt):
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake2[1:]:
            return True
        if pt in self.snake:
            return True
        for bomb in self.bombs:
            if pt == bomb:
                return True
        return False

    def _move_hardcoded_snake(self):
        """
        Modified greedy strategy:
        - Computes all safe moves.
        - With a 30% chance, selects a random safe move (dumbing it down slightly).
        - Otherwise, chooses the move that minimizes the Manhattan distance to the food.
        """
        possible_actions = [0, 1, 2]  # 0: straight, 1: right, 2: left
        safe_actions = []
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction2)
        for action in possible_actions:
            if action == 0:
                new_dir = clock_wise[idx]
            elif action == 1:
                new_dir = clock_wise[(idx + 1) % 4]
            else:
                new_dir = clock_wise[(idx - 1) % 4]
            x = self.head2.x
            y = self.head2.y
            if new_dir == Direction.RIGHT:
                x += BLOCK_SIZE
            elif new_dir == Direction.LEFT:
                x -= BLOCK_SIZE
            elif new_dir == Direction.DOWN:
                y += BLOCK_SIZE
            elif new_dir == Direction.UP:
                y -= BLOCK_SIZE
            new_pt = Point(x, y)
            if not self._is_collision_snake2(new_pt):
                dist = abs(self.food.x - new_pt.x) + abs(self.food.y - new_pt.y)
                safe_actions.append((action, new_dir, new_pt, dist))

        if not safe_actions:
            action = 0
            new_dir = clock_wise[idx]
            x = self.head2.x
            y = self.head2.y
            if new_dir == Direction.RIGHT:
                x += BLOCK_SIZE
            elif new_dir == Direction.LEFT:
                x -= BLOCK_SIZE
            elif new_dir == Direction.DOWN:
                y += BLOCK_SIZE
            elif new_dir == Direction.UP:
                y -= BLOCK_SIZE
            best_new_head = Point(x, y)
        else:
            if random.random() < 0.3:
                chosen = random.choice(safe_actions)
            else:
                chosen = min(safe_actions, key=lambda x: x[3])
            action, new_dir, best_new_head, _ = chosen
            self.direction2 = new_dir

        self.head2 = best_new_head
        self.snake2.insert(0, self.head2)
        if self.head2 == self.food:
            self._place_food()
        else:
            self.snake2.pop()
        if self._is_collision_snake2(self.head2):
            self._init_hardcoded_snake()

    def _update_ui(self, speed=SPEED):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, (pt.x + 4, pt.y + 4, 12, 12))
        for bomb in self.bombs:
            pygame.draw.rect(self.display, GREEN, (bomb.x, bomb.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, (self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        for pt in self.snake2:
            pygame.draw.rect(self.display, YELLOW, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(speed)

    def _get_observation(self):
        dx_food = (self.food.x - self.head.x) / float(self.w)
        dy_food = (self.food.y - self.head.y) / float(self.h)
        if self.bombs:
            nearest_bomb = min(self.bombs, key=lambda b: abs(b.x - self.head.x) + abs(b.y - self.head.y))
            dx_bomb = (nearest_bomb.x - self.head.x) / float(self.w)
            dy_bomb = (nearest_bomb.y - self.head.y) / float(self.h)
        else:
            dx_bomb, dy_bomb = 0.0, 0.0
        dir_r = 1 if self.direction == Direction.RIGHT else 0
        dir_l = 1 if self.direction == Direction.LEFT  else 0
        dir_u = 1 if self.direction == Direction.UP    else 0
        dir_d = 1 if self.direction == Direction.DOWN  else 0
        danger_straight = 1 if self._will_collide_if_action(0) else 0
        danger_right    = 1 if self._will_collide_if_action(1) else 0
        danger_left     = 1 if self._will_collide_if_action(2) else 0
        snake_len = len(self.snake) / 100.0
        env_snake_dir_normalized = 0.0
        state = np.array([
            dx_food, dy_food,
            dx_bomb, dy_bomb,
            dir_r, dir_l, dir_u, dir_d,
            0.0, 0.0,
            danger_straight, danger_right, danger_left,
            snake_len,
            env_snake_dir_normalized
        ], dtype=np.float32)
        return state

    def _will_collide_if_action(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
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

    def _save_video(self, frames, score):
        """
        Saves the given frames to an MP4 file and returns the filename.
        """
        if not frames:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultimate_final_snake_videos/snake_score_{score}_{timestamp}.mp4"
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        log_score(f"Video saved: {filename}")
        return filename

if __name__ == "__main__":
    env = SnakeEnv(render_mode=True)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_snake/",
        learning_rate=LEARNING,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=CLIP,
        ent_coef=ENT_COEF
    )

    model.learn(total_timesteps=TIME_STEP)
    model.save("ppo_ultimate_final")

    # Print out the best-case video at the end
    raw_env = env.envs[0]
    if raw_env.best_video_filename is not None:
        print(f"\nBest video file: {raw_env.best_video_filename} (score = {raw_env.best_score})\n")
    else:
        print("\nNo best video recorded.\n")