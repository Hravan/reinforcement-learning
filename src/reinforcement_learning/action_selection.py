from __future__ import annotations
import random

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from reinforcement_learning.agent import Agent


class UCB:
    def __init__(self, exploration_coefficient: float):
        self.exploration_coefficient = exploration_coefficient
    
    def __call__(self, agent: Agent):
        criterion_values = []
        for action_index, _ in enumerate(agent.actions):
            n_selected = agent.n_selected(action_index)

            if n_selected == 0:
                criterion_values.append(float('inf'))
                continue

            current_estimate = agent.reward_estimates[action_index]
            timestep = agent.n_choices
            criterion_value = current_estimate + self.exploration_coefficient * (np.log(timestep) / n_selected) ** (1/2)
            criterion_values.append(criterion_value)
        return np.argmax(criterion_values)
    
    def update(self, *args, **kwargs):
        pass


class GradientBandit:
    def __init__(self, alpha: float, baseline: float, n_actions: int | None = None):
        self.alpha = alpha
        self.baseline = baseline
        self.action_preferences = [0.0 for _ in range(n_actions)] if n_actions is not None else None
    
    def __call__(self, agent):
        if self.action_preferences is None:
            self.action_preferences = [0.0 for _ in agent.actions]
        action_choice = random.choices(range(len(agent.actions)), weights=self.action_probabilities, k=1)
        return action_choice[0]
    
    @property
    def action_probabilities(self):
        preferences_exponentiated = [np.exp(preference) for preference in self.action_preferences]
        denominator = sum(preferences_exponentiated)
        probabilities = [preference / denominator for preference in preferences_exponentiated]
        return probabilities
    
    def update(self, action_index: int, reward: float):
        probabilities = self.action_probabilities
        self.action_preferences[action_index] += self.alpha * (reward - self.baseline) * (1 - probabilities[action_index])
        for i, _ in enumerate(self.action_preferences):
            if i != action_index:
                self.action_preferences[i] += -self.alpha * (reward - self.baseline) * probabilities[i]


class RandomActionSelection:
    def __call__(self, agent):
        return random.randrange(len(agent.actions))
    
    def update(self, *args, **kwargs):
        pass


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, agent):
        if random.random() > 1 - self.epsilon:
            action_index = random.randrange(len(agent.actions))
        else:
            action_index = np.argmax(agent.reward_estimates)
        return action_index
    
    def update(self, *args, **kwargs):
        pass