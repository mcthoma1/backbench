---
layout: default
title: Final Report
---

## Video:
[![Video Title]]("")

## Project Summary:


## Aproaches
We compare a baseline (PPO with a single bomb) and more advanced approaches, including QR-DQN (static vs. dynamic reward) and DQN (dynamic reward). The environment complexity includes a dynamic number of bombs and an additional hardcoded competitive snake.

### Baseline: PPO with a Single Bomb

### QRDQN (Quantile-Regression Deep Q-Network):
Quantile Regression Deep Q-Network (QRDQN) extends the standard DQN by learning a distribution of future rewards rather than a single expected value. This approach provides a more robust way to estimate action-value functions, as it captures the uncertainty in possible returns. Instead of predicting *Q(s,a)* as a scalar, QRDQN predicts multiple quantiles of the return distribution, allowing it to model the variability in outcomes. This is particularly useful in environments with stochastic elements, such as our dynamic Snake game, where the number of bombs increases based on performance and where a second competitive snake introduces unpredictable behavior.

In our implementation, we trained QRDQN under two different reward schemes: **static rewards** and ***dynamic rewards**. The s**static reward** model assigns fixed values for each event: +10 for eating food and −10 for penalty (i.e. collision, bomb), regardless of the snake’s length. This ensures a straightforward, predictable learning objective, making it easier to tune hyperparameters. However, a potential downside is that the agent may learn overly conservative behavior, as it has no incentive to take strategic risks for larger rewards. 

In contrast, the **dynamic reward** model scales the food reward with the snake’s length, rewarding longer survival and more complex maneuvers. While this encourages more aggressive strategies, it also introduces additional variability in training, as reward magnitudes change over time. This can make hyperparameter tuning more challenging, particularly for learning rate adjustments and exploration strategies. However, QRDQN’s distributional nature makes it well-suited to handle these variations by learning from multiple possible outcomes rather than relying solely on a single expected value.

However, it's to be noted that we used the same parameters as the dqn for this model, so it's essentially dqn but improved. 

## DQN (Deep Q-Network):
Deep Q-Network (DQN) is a reinforcement learning algorithm that extends Q-learning by using a neural network to approximate the Q-values instead of maintaining a Q-table. This allows DQN to scale to high-dimensional state spaces, making it well-suited for environments like Snake, where the state is represented as a 15-dimensional feature vector rather than a discrete grid. DQN learns the expected cumulative reward for each action and updates its Q-values using experience replay and a target network to stabilize training.

In our implementation, DQN was trained exclusively with a dynamic reward scheme. Unlike a static reward system, where food collection always yields a fixed reward, dynamic rewards scale with the snake’s length. This encourages longer survival and riskier strategies as the game progresses. Specifically, the agent receives:
- +10×snake length
- +10×snake length for eating food.
− −10 for collisions (wall, self, or second snake).
− −7 for hitting a bomb.
- A small time penalty to discourage unnecessary delays.

Another key aspect of training was the increasing number of bombs based on the agent’s success. This progressively escalated the difficulty, forcing the model to generalize its policy to handle more obstacles over time. Additionally, a second snake was introduced as a hardcoded competitor, making food collection more competitive. This added another level of environmental complexity.

The disadvantage of DQN is that it only estimates a single expected reward. This makes it less robust in environments with high variability, such as ours with dynamic rewards, bombs, and a second snake. Despite this, ut still performed really well and was still comparable to QRDQN. 
```
for episode in range(max_episodes):
    state = env.reset()
    done = False
    while not done:
        # Take action, observe next state and reward
        next_state, reward, done = env.step(action)

        # Store transition in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Sample batch from replay buffer
        if len(replay_buffer) > batch_size:
            batch = sample(replay_buffer, batch_size)

            # Compute target Q-values
            Q_target = reward + gamma * max(Target_network(next_state))

            # Compute loss and update Q-network
            loss = MSE(Q_network(state, action), Q_target)
            backpropagate(loss)  # Update Q-network
```