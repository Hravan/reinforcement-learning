from typing import TypeAlias
import numpy as np

Reward: TypeAlias = float

class Action:
    def __init__(self, value: float, std: float):
        self.value = value
        self.std = std
    
    def perform(self) -> Reward:
        return np.random.normal(self.value, self.std)
    
    @classmethod
    def gaussian(cls, mean, std):
        value = np.random.normal(mean, std)
        return cls(value, std)
