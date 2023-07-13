This is a fork from : https://github.com/vwxyzjn/cleanrl/tree/master/cleanrl

# LunarLander-v2 Experiments - Gym - Pytorch

These experiments feature discrete action space and (multi-)discrete observation space

## PPO - Proximal Policy Optimization

![Training](/training.png)

![Experiments records](/rl-video-episodes.gif)


## PPO - LSTM

## PPO - CNN - LSTM

Use a custom gym wrapper to generate sequence of grayscale images for the episode

```python
class StateToRGBImage(gym.ObservationWrapper):
    def __init__(self, env, width=300, height=200):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

    def observation(self, state):
        img = self.env.render(mode='rgb_array')
        img_resized = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img_resized


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = StateToRGBImage(env)
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
```

## PPO Multi-Discrete Observation

Split observation from [ X, X, X, X, X, X, X, X ] to  [ [X, X], [X, X], [X, X], [X, X] ].

Tensor activations are then added together before getting passed to actor and critic layers.