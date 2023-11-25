# Import necessary libraries and modules
import numpy as np
import gym
from configs import BtcMarketEnvConfig
from gym.spaces import Tuple, Discrete, Box

import json
from tqdm import tqdm

import random

# Define the BtcMarketEnv class which is a subclass of gym.Env
class BtcMarketEnv(gym.Env):
    # Initialize the class with data, prices, and a config
    def __init__(self, data,  prices, config=None):
        # Set config to BtcMarketEnvConfig if not provided
        self.config = config #if config is not None else BtcMarketEnvConfig()
        # Initialize environment
        self._initialize_environment(prices, self.config)
        # Set other properties
        self.data=data
        self.extra_features = 5
        self.num_columns = data.shape[1] + self.extra_features  
        self.window_size = config.window_size
        # Define observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.config.batch_size, self.window_size, self.num_columns), dtype=np.float32)
        # Initialize data buffer
        self.data_buffer = np.zeros((len(self.data) + self.config.window_size, self.num_columns))
        # Fill in data buffer with data
        self.data_buffer[:len(self.data), :-self.extra_features] = self.data
        # Reset environment
        self.reset(self.starting_step)

    # Method to initialize environment
    def _initialize_environment(self,  prices,  config):
        # Set various properties
        self.o, self.h, self.l, self.c = prices[:4]
        self.ts = prices[4]
        self.max_steps= config.max_steps
        self.starting_step = self.config.starting_step
        self.ending_step= self.starting_step + self.max_steps
        # Define action space
        self.action_space = Discrete(4)  # Actions: Buy, sell, hold, close
        self.current_step = self.current_ts = self.observation = None

    # Method to reset environment
    def reset(self,start):
        # Set various properties
        self.current_open = self.o[start]
        self.current_high = self.h[start]
        self.current_low = self.l[start]
        self.current_close = self.c[start]
        self.current_step = start 
        self.current_ts = self.ts[start]
        # Roll the data buffer
        self.data_buffer = np.roll(self.data_buffer, shift=-self.config.max_steps, axis=0)
        # Get next observation
        self.observation = self._get_next_observation() 
        return self.observation

    # Method to get next observation
    def _get_next_observation(self):
        start = self.current_step % self.config.max_steps
        end = start + self.config.window_size
        obs = self.data_buffer[start:end]
        return obs

    # Method to update step features
    def update_step_features(self, step, features):
        self.data_buffer[step, -len(features):] = features

    # Method to update prices and timestamps
    def update_prices_and_ts(self, o, h, l, c, ts):
        self.current_open = o
        self.current_high = h
        self.current_low = l
        self.current_close = c
        self.current_ts = ts

    # Method to perform a step in the environment
    def step(self):
        # Update prices and timestamp
        self.update_prices_and_ts(
            self.o[self.current_step],
            self.h[self.current_step],
            self.l[self.current_step],
            self.c[self.current_step],
            self.ts[self.current_step]
        )
        # Increase current step
        if self.current_step < len(self.data) - 1:    # Check if current step is less than the last index
            self.current_step += 1
        if self.current_step >= len(self.data):
            # Roll the data buffer if the end of the data is reached
            self.data_buffer = np.roll(self.data_buffer, shift=-self.config.max_steps, axis=0)
        # Get next observation
        self.observation = self._get_next_observation()
        return self.observation
