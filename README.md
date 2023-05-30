This is a fork from : https://github.com/vwxyzjn/cleanrl/tree/master/cleanrl

# LunarLander-v2 Experiments - Gym - Pytorch

These experiments feature discrete action space and (multi-)discrete observation space

## PPO

![Training](/training.png)

## PPO - LSTM

## PPO - CNN - LSTM

Use a custom gym wrapper to generate grayscale image of environment

## PPO Multi-Discrete Observation

Split observation from [ X, X, X, X, X, X, X, X ] to  [ [X, X], [X, X], [X, X], [X, X] ].

Tensor activations are then added together before getting passed to actor and critic layers.