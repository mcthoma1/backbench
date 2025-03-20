---
layout: default
title: Final Report
---

## Project Summary:
The main goal of our project was to find how well Proximal Policy Optimization (PPO) and Deep Q-Network (DQN) handles a evolving and increasing complex environment. We did this by using a classic snake game as the scenario. We chose game as our scenario for reinforcement learning because this is one of our childhood classics and we always wondered if it would be possilbe to have the snake reach the length that covers the whole board. In order to add complexity to see how much we could push the learning algorithms, we modified the environement of the traditional snake game: we had the snake create efficient paths to the food, grow larger, avoid colliding with bombs and also compete with a secondary competitive snake who would compete against the agent for the food and could kill the agent snake if it is bigger that it. 

In the modified environment, we would increase the number of bombs by 1 (maximum limit of 25) every time the snake would eat the food and also increase the length of the snake which increased the risk of self collision. Any self collision or collision with the bombs would end the episode which increases the difficulty of the game as the game progresses and the snake grows larger. The secondary snake also served as an obstacle for the agent. The snake was hardcoded with a greedy approach who would compete against the agent snake for the food. If the second snake was bigger than the agent snake then it will kill the agent snake if the agent snake collided with it. This would forced the agent to create an efficient path to the food to reach it before the second snake and eventually get bigger than it so that collisions would kill it. 

The constant evolving dynamics of our environment make the task of getting ott he food nad getting larger non trivial. This is where radiaional algorithsm would fail as they wouldn't be able to adapt or generalize the changing envrionemtn due to their pre defined rules. This leads to the necessity of adaptive solutions. 

This is why reinforcement learning algorithms are necessary due to their ability to interact and learn from their environment and adjust their strategies and decisions rather than relying on predefined rules. PPO's incremental decision making policy and DQN's ability to use past interactions with the environment to make future decisions allow them to navgivate and efficiently play the evolving and challenging environment of the snake game. Our approach showcases the ability of these algirthms to excel in complex scenarios and also its capability in solving real world tasks such as autonomous vehicle navigation. 


<img src="image.png" alt="Project Screenshot" width="400"/>

Figure 1.0: Picture of how the snake game worked
- Blue Snake: Agent
- Yellow Snake: Second Snake (competitor)
- Green squares: bombs
- Red squares: food


## Video:
[![Video Title](https://img.youtube.com/vi/KmY3JyX5_js/0.jpg)](https://youtu.be/KmY3JyX5_js)



## Aproaches
We compare a baseline (PPO with a single bomb) and more advanced approaches, including QR-DQN (static vs. dynamic reward) and DQN (dynamic reward). The environment complexity includes a dynamic number of bombs and an additional hardcoded competitive snake.

### Baseline: PPO with a Single Bomb

## DQN (Deep Q-Network):
Deep Q-Network (DQN) is a reinforcement learning algorithm that extends Q-learning by using a neural network to approximate the Q-values instead of maintaining a Q-table. This allows DQN to scale to high-dimensional state spaces, making it well-suited for environments like Snake, where the state is represented as a 15-dimensional feature vector rather than a discrete grid. DQN learns the expected cumulative reward for each action and updates its Q-values using experience replay and a target network to stabilize training.

In our implementation, DQN was trained exclusively with a dynamic reward scheme. Unlike a static reward system, where food collection always yields a fixed reward, dynamic rewards scale with the snake’s length. This encourages longer survival and riskier strategies as the game progresses. Specifically, the agent receives:
- +10×snake length
- +10×snake length for eating food.
- -10 for collisions (wall, self, or second snake).
- (−7) for hitting a bomb.
- A small time penalty to discourage unnecessary delays.

Another key aspect of training was the increasing number of bombs based on the agent’s success. This progressively escalated the difficulty, forcing the model to generalize its policy to handle more obstacles over time. Additionally, a second snake was introduced as a hardcoded competitor, making food collection more competitive. This added another level of environmental complexity.

The disadvantage of DQN is that it only estimates a single expected reward. This makes it less robust in environments with high variability, such as ours with dynamic rewards, bombs, and a second snake. Despite this, ut still performed really well and was still comparable to QRDQN. 

### QRDQN (Quantile-Regression Deep Q-Network):
Quantile Regression Deep Q-Network (QRDQN) extends the standard DQN by learning a distribution of future rewards rather than a single expected value. This approach provides a more robust way to estimate action-value functions, as it captures the uncertainty in possible returns. Instead of predicting *Q(s,a)* as a scalar, QRDQN predicts multiple quantiles of the return distribution, allowing it to model the variability in outcomes. This is particularly useful in environments with stochastic elements, such as our dynamic Snake game, where the number of bombs increases based on performance and where a second competitive snake introduces unpredictable behavior.

***It's to be noted that we used the same parameters for QRDQN and DQN so it's essentially dqn but improved.***

### Proximal Policy Optimization (PPO):
Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that refines policy learning through a clipped objective function, balancing exploration and exploitation. It is well-suited for our dynamic Snake game environments, bomb count scales with performance and a competitive snake introduces unpredictability, since it ensures stability and adaptability while maintaining sample efficiency.

In our implementation, we trained QRDQN and PPO under two different reward schemes: **static rewards** and **dynamic rewards**. The **static reward** model assigns fixed values for each event: +10 for eating food and −10 for penalty (i.e. collision, bomb), regardless of the snake’s length. This ensures a straightforward, predictable learning objective, making it easier to tune hyperparameters. However, a potential downside is that the agent may learn overly conservative behavior, as it has no incentive to take strategic risks for larger rewards. 

In contrast, the **dynamic reward** model scales the food reward with the snake’s length, rewarding longer survival and more complex maneuvers. While this encourages more aggressive strategies, it also introduces additional variability in training, as reward magnitudes change over time. This can make hyperparameter tuning more challenging, particularly for learning rate adjustments and exploration strategies. For PPO, this is a reliable aspect since the algorithm relies heavily on the reward policy. 

```
on step(action):
    store the expereience in the buffer
    
model = QR/DQN(
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

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_snake/",
    learning_rate=0.0003, 
    n_steps=4096, 
    batch_size=256, 
    n_epochs=10,
    gamma=0.9, 
    gae_lambda=0.95, 
    clip_range=0.2, 
    ent_coef=0.3
)

```
## Evaluation

## PPO
In our PPO approach, we observed that the model’s improvement plateaued in static reward structure depending on the environment (*see tensorboard chart below). To address this, we did some experiments to find a fair and compact reward structure suitable for all environment, with a set of tuned hyperparameters.

### Environments Tested
#### 1 Bomb Environment  
<img src="./image/1b_graph.png" alt="1 Bomb Graph" width="300"/><br/>
<img src="./image/1b_chart.png" alt="1 Bomb Graph" width="300"/>

#### 10 Bombs Environment  
<img src="./image/10b_graph.png" alt="10 Bombs Graph" width="300"/><br/>
<img src="./image/10b_chart.png" alt="10 Bombs Graph" width="300"/>

#### 25 Bombs Environment  
<img src="./image/25b_graph.png" alt="25 Bombs Graph" width="300"/><br/>
<img src="./image/25b_chart.png" alt="25 Bombs Graph" width="300"/>

#### Performance Chart Matrix  
<img src="./image/chart.png" alt="Performance Chart" width="300"/>


## Resources Used:
- https://www.youtube.com/watch?v=L8ypSXwyBds : Explained how reinforcement learning can be used to play snake game and inspired us on how to structure teh environement and objects
- https://github.com/patrickloeber/snake-ai-pytorch : guided on to train agents in a snake game using reinforcement learning through pytorch
- https://medium.com/deeplearningmadeeasy/simple-ppo-implementation-e398ca0f2e7c : Taught us the basics of PPO and how it is used to optimize policy updates when training 
- https://sohum-padhye.medium.com/building-a-reinforcement-learning-agent-that-can-play-rocket-league-5df59c69b1f5: guided us on how to train agents in complex environments and high stochasity in deep reinforcement learning models
- https://papers-100-lines.medium.com/reinforcement-learning-ppo-in-just-100-lines-of-code-1f002830cff4: explained to us the core concept of PPO through simple implementations of the policy 
- https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c: taught us how we can use the reward system in the snake game to improve the performance of the agent 
- https://stable-baselines3.readthedocs.io/en/v0.11.1/modules/dqn.html#notes 
- https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html
- https://github.com/Stable-Baselines-Team/stable-baselines3-contrib?tab=readme-ov-file
- https://github.com/araffin/rlss23-dqn-tutorial
- *Stable-Baselines3*: Used for implementing PPO, DQN and managing policy updates.
- *SB3_contrib*: Used for QR-DQN.
- *Gymnasium*: Used the API to build the environment for the snake game.
- *PyTorch*: used for deep learning of the PPO approach.
- *Matplotlib*: Helped visualize training performance and results.
- *TensorBoard*: Used for logging and tracking model improvements.
- We used AI as a valuable learning source. We used AI models like chatGPT, research papers, and documentations to understand difficult concepts in reinforcement learning, PPO, and DQN/QRDQN, fix errors and issues with our implementation, and improve our approach to solving the problems. It assisted us by allowing us to compare our ideas with other ideas and approaches, explaining complex ideas, and validate our implementations and appraoches. This allowed us to learn and understand how we could better tune our hyperparameters, structure our environments and other parts of the model, and make good decisions on our strategy in the project.  