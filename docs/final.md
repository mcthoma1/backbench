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

### QRDQN:
Quantile Regression Deep Q-Network (QRDQN) extends the standard DQN by learning a distribution of future rewards rather than a single expected value. This approach provides a more robust way to estimate action-value functions, as it captures the uncertainty in possible returns. Instead of predicting *Q(s,a)* as a scalar, QRDQN predicts multiple quantiles of the return distribution, allowing it to model the variability in outcomes. This is particularly useful in environments with stochastic elements, such as our dynamic Snake game, where the number of bombs increases based on performance and where a second competitive snake introduces unpredictable behavior.

In our implementation, we trained QRDQN under two different reward schemes: **static rewards** and ***dynamic rewards**. The s**static reward** model assigns fixed values for each event: +10 for eating food and −10 for penalty (i.e. collision, bomb), regardless of the snake’s length. This ensures a straightforward, predictable learning objective, making it easier to tune hyperparameters. However, a potential downside is that the agent may learn overly conservative behavior, as it has no incentive to take strategic risks for larger rewards. 

In contrast, the **dynamic reward** model scales the food reward with the snake’s length, rewarding longer survival and more complex maneuvers. While this encourages more aggressive strategies, it also introduces additional variability in training, as reward magnitudes change over time. This can make hyperparameter tuning more challenging, particularly for learning rate adjustments and exploration strategies. However, QRDQN’s distributional nature makes it well-suited to handle these variations by learning from multiple possible outcomes rather than relying solely on a single expected value.