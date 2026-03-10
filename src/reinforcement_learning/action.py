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
