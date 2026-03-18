from typing import TypeAlias
import numpy as np

Reward: TypeAlias = float

class Action:
    def __init__(self, value: float, std: float, stationary=True):
        self.value = value
        self.std = std
        self.stationary = stationary
    
    def perform(self) -> Reward:
        reward = np.random.normal(self.value, self.std)
        if not self.stationary:
            self.value = self.value + np.random.normal(0, 0.01)
        return reward
    
    @classmethod
    def gaussian(cls, mean, std, **kwargs):
        value = np.random.normal(mean, std)
        return cls(value, std, **kwargs)
    
    def __repr__(self):
        return f'Action(value={self.value}, std={self.std}, stationary={self.stationary})'


class Experience:
    def __init__(self):
        self.action_history = []
        self.sum_rewards = 0
    
    def update(self, action, reward):
        self.action_history.append(action)
        self.sum_rewards += reward
    
    def n_selected(self, action_index):
        return sum(1 for action in self.action_history if action == action_index)
    
    @property
    def last_action(self):
        return self.action_history[-1]
    
    def n_selected_last_action(self):
        return self.n_selected(self.last_action)
    
    def __len__(self):
        return len(self.action_history)
    
    @property
    def mean_reward(self):
        return self.sum_rewards / len(self.action_history)
