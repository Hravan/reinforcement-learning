import random
from types import MethodType

import numpy as np

from reinforcement_learning.action import Experience
from reinforcement_learning.action_selection import RandomActionSelection, EpsilonGreedy


class Agent:
    def __init__(self, *actions, initial_reward_value=0, step_size=None, action_selection_method=RandomActionSelection()):
        self.actions = actions
        # TODO: make reward estimates part of the reward selection and estimation class (or two classes)
        self.reward_estimates = [initial_reward_value for _ in self.actions]
        self.experience = Experience()

        if step_size is not None:
            self.step_size = MethodType(step_size, self)
        if action_selection_method is not None:
            self.select_action = MethodType(action_selection_method, self)
    
    def act(self):
        action_index = self.select_action()
        reward = self.actions[action_index].perform()
        self.experience.update(action_index, reward)
        self.select_action.update(action_index, reward)

        current_estimate = self.reward_estimates[action_index]
        self.reward_estimates[action_index] = current_estimate + self.step_size() * (reward - current_estimate)
    
    @property
    def optimal_action(self):
        return np.argmax([action.value for action in self.actions])
    
    @property
    def mean_reward(self):
        return self.experience.mean_reward
    
    def step_size(self):
        return 1 / self.experience.n_selected_last_action()
    
    def n_selected(self, action_index):
        return self.experience.n_selected(action_index)

    @property
    def n_choices(self):
        '''Number of choices already made by the agent.'''
        return len(self.experience)


class EpsilonGreedyAgent(Agent):
    def __init__(self, *actions, epsilon=0, **kwargs):
        super().__init__(*actions, action_selection_method=EpsilonGreedy(epsilon), **kwargs)


class ConstantStepSize:
    def __init__(self, step_size):
        self.step_size = step_size
    
    def __call__(self, agent):
        return self.step_size
