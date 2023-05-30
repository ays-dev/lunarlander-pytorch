This is a fork from : https://github.com/vwxyzjn/cleanrl/tree/master/cleanrl

# LunarLander-v2 Experiments - Pytorch

These experiments feature discrete action space and (multi-)discrete observation space

## PPO

![Training](/training.png)

## PPO - LSTM

## PPO - CNN - LSTM

Use a custom wrapper to generate grayscale image of environment

## PPO Multi-Discrete Observation

Split observation from [ x, x, x, x, x, x, x, x ] to  [ [x, x], [x, x], [x, x], [x, x] ].

Tensor activations are then added together before getting passed to actor and critic layers.