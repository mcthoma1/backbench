---
layout: default
title: Status
---

## Project Summary:
check one two 



## Evaluation:
📈 Quantitative Analysis
There are two key factors for quantitatively evaluating our model: survival time and average score. All of our agents, trained with different hyperparameters, demonstrate varying average scores (rewards) and survival times. The following graph shows the progress of training episodes: Reward trends over multiple training episodes.
![Screenshot](screenshots/chart.png)  

🧐 Qualitative Analysis
Through the observation, we discovered that none of our models significantly increased their average score between 500k and 1M timesteps. To facilitate better analysis, we decide to limit for future training within 1M timesteps to focus on meaningful improvements in performance.