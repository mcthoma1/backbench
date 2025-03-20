# backbench

## Video:
[![Video Title](https://img.youtube.com/vi/KmY3JyX5_js/0.jpg)](https://youtu.be/KmY3JyX5_js)



## Project Summary:
The main goal of our project was to find how well Proximal Policy Optimization (PPO) and Deep Q-Network (DQN) handles a evolving and increasing complex environment. We did this by using a classic snake game as the scenario. We chose game as our scenario for reinforcement learning because this is one of our childhood classics and we always wondered if it would be possilbe to have the snake reach the length that covers the whole board. In order to add complexity to see how much we could push the learning algorithms, we modified the environement of the traditional snake game: we had the snake create efficient paths to the food, grow larger, avoid colliding with bombs and also compete with a secondary competitive snake. 

In the modified environment, we would increase the number of bombs by 1 (maximum limit of 25) every time the snake would eat the food and also increase the length of the snake. This increased the chance of self collision. Any self collision or collision with the bombs would end the episode. This might seem easy at the beginning it got more and more difficult for the agent the number of bombs increaased and the size of the snake also increased.

The constant evolving dynamics of our environment make the task of getting the food and getting larger non trivial. This is where hardcoded algorithms would fail as they wouldn't be able to adapt or generalize the changing envrionemtn due to their pre defined rules. This leads to the necessity of adaptive solutions. 

This is why reinforcement learning algorithms are necessary due to their ability to interact and learn from their environment and adjust their strategies and decisions rather than relying on predefined rules. PPO's incremental decision making policy and DQN's ability to use past interactions with the environment to make future decisions allow them to navgivate and efficiently play the evolving and challenging environment of the snake game. Our approach showcases the ability of these algirthms to excel in complex scenarios and also its capability in solving real world tasks such as autonomous vehicle navigation. 

<img src="image.png" alt="Project Screenshot" width="300"/>
 
 Figure 1.0: Picture of how the snake game worked
 - Blue Snake: Agent
 - Yellow Snake: Second Snake (competitor)
 - Green squares: bombs
 - Red squares: food