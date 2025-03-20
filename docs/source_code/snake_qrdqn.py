import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
from enum import Enum
from sb3_contrib import QRDQN  
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch
import os
import cv2
from datetime import datetime

TIME_COMSUME = 0.01
TIME_STEP = 1000000  # 1M timesteps for training
BLOCK_SIZE = 20
SPEED = float('inf')  
RECORD_SPEED = 10000
LEARNING = 0.0003  # PPO alsso had 0.0003, but for DQN/QR-DQN.

WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # bombs
YELLOW = (255, 255, 0)  

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

def log_score(message, log_file="best_ppo_3b.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {message}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class SnakeEnv(gym.Env):
    """
    Single-agent snake environment:
      - 640x480 board
      - Discrete(3) actions: 0=straight,1=turn-right,2=turn-left
      - 15-dim observation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, w=640, h=480, render_mode=False):
        super(SnakeEnv, self).__init__()
        self.w = w
        self.h = h
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(15,), dtype=np.float32
        )

        # initalizise pygame
        pygame.init()
        self.font = pygame.font.SysFont('arial', 25)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake DQN')
        self.clock = pygame.time.Clock()

        os.makedirs("3b_snake_videos", exist_ok=True)

        self.num_bombs = 1  
        self.goal_score = 1
        self.episode_scores = []
        self.recording = False
        self.frames = []
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        if not hasattr(self, "episode_scores"):
            self.episode_scores = []
        
        if hasattr(self, "score"):
            self.episode_scores.append(self.score)
            if self.score >= self.goal_score:
                log_score(f"Goal reached with score: {self.score}, bomb++ {self.num_bombs + 1}")
                self.goal_score = self.score + 1
                self.num_bombs += 1
            
            if len(self.episode_scores) % 30 == 0:
                avg_score = sum(self.episode_scores[-30:]) / 30
                highest_score = max(self.episode_scores)
                log_score(f"Last 30 Episodes Avg Score: {avg_score:.2f}, Highest Score: {highest_score}")

            if self.recording and self.frames:
                self._save_video()
                self.recording = False
                self.frames = []

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

        self.score = 0
        self.frame_iteration = 0
        
        self._place_food()

        self.bombs = []
        self._place_bomb()
        
        self._init_hardcoded_snake()

        return self._get_observation(), {}

    def step(self, action):
        self.frame_iteration += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Move main snake
        self._move(action)
        self.snake.insert(0, self.head)

        # Collision?
        if self._is_collision(self.head):
            terminated = True
            reward = -10

        # Time penalty
        reward -= TIME_COMSUME

        if not terminated and self.frame_iteration > 100 * len(self.snake):
            truncated = True
            reward = -10

        # Eat food
        if not terminated and not truncated:
            if self.head == self.food:
                self.score += 1
                reward = 10 * len(self.snake)
                self._place_food()
            else:
                self.snake.pop()

        # Check bombs
        if not terminated and not truncated:
            for bomb in self.bombs:
                if self.head == bomb:
                    terminated = True
                    reward = -7
                    break

        if self.score >= (self.goal_score - 5) and self.score >= 2 and not self.recording:
            self.recording = True
            self.frames = []

        if self.recording:
            self._update_ui(RECORD_SPEED)
            frame = self.render()
            self.frames.append(frame)

        if (terminated or truncated) and self.recording:
            if self.score >= self.goal_score:  
                self._save_video()
            self.recording = False
            self.frames = []

        self._move_hardcoded_snake()

        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, {}

    def render(self, mode="rgb_array"):
        frame = pygame.surfarray.array3d(self.display)
        frame = np.transpose(frame, (1, 0, 2))  # (H, W, C)
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
        while (self.food in self.snake or (hasattr(self, 'bombs') and self.food in self.bombs)):
            x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)

    def _place_bomb(self):
        self.bombs = []
        for _ in range(self.num_bombs):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                bomb = Point(x, y)
                while (bomb in self.snake or (hasattr(self, 'food') and bomb == self.food)):
                    x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
                    y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
                    bomb = Point(x, y)
                if bomb not in self.snake and bomb != self.food:
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
        possible_actions = [0, 1, 2]  # 0: straight, 1: right, 2: left
        best_action = None
        best_distance = float('inf')
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
                if dist < best_distance:
                    best_distance = dist
                    best_action = action
        if best_action is None:
            best_action = 0
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
            if best_action == 0:
                new_dir = clock_wise[idx]
            elif best_action == 1:
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
            best_new_head = Point(x, y)
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
            pygame.draw.rect(self.display, BLUE2, (pt.x+4, pt.y+4, 12, 12))
        
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

        # nearest bomb
        if self.bombs:
            nearest_bomb = min(self.bombs, key=lambda b: abs(b.x - self.head.x) + abs(b.y - self.head.y))
            dx_bomb = (nearest_bomb.x - self.head.x) / float(self.w)
            dy_bomb = (nearest_bomb.y - self.head.y) / float(self.h)
        else:
            dx_bomb = 0.0
            dy_bomb = 0.0
        
        dir_r = 1 if self.direction == Direction.RIGHT else 0
        dir_l = 1 if self.direction == Direction.LEFT  else 0
        dir_u = 1 if self.direction == Direction.UP    else 0
        dir_d = 1 if self.direction == Direction.DOWN  else 0
        
        # Danger checks
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

    def _save_video(self):
        if not self.frames:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"3b_snake_videos/snake_score_{self.score}_{timestamp}.mp4"
        
        height, width, layers = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
        
        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        log_score(f"Video saved: {filename}")

if __name__ == "__main__":
    env = SnakeEnv(render_mode=True)
    env = Monitor(env)                 # Logs episodes
    env = DummyVecEnv([lambda: env])   # Vectorize

 
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = QRDQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_snake/",
        learning_rate=0.0003,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        target_update_interval=10000, 
    )

    # Train
    model.learn(total_timesteps=TIME_STEP)
    model.save("best_qrdqn_3b")