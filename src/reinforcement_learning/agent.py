import random
import numpy as np


class Agent:
    def __init__(self, *actions, initial_reward_value=0):
        self.actions = actions
        self.reward_estimates = [initial_reward_value for _ in self.actions]
        self.action_history = []
        self.sum_rewards = 0
    
    def act(self):
        action_index = self.choice()
        self.action_history.append(action_index)
        reward = self.actions[action_index].perform()
        self.sum_rewards += reward

        current_estimate = self.reward_estimates[action_index]
        n_choices = sum(1 for action in self.action_history if action == action_index)
        self.reward_estimates[action_index] = current_estimate + (1 / n_choices) * (reward - current_estimate)
    
    def choice(self):
        action_index = random.randrange(len(self.actions))
        return action_index
    
    @property
    def optimal_action(self):
        return np.argmax(action.value for action in self.actions)
    
    @property
    def mean_reward(self):
        return self.sum_rewards / len(self.action_history)
    

class EpsilonGreedyAgent(Agent):
    def __init__(self, *actions, epsilon=0):
        super().__init__(*actions)
        self.epsilon = epsilon

    def choice(self):
        if random.random() > 1 - self.epsilon:
            action_index = random.randrange(len(self.actions))
        else:
            action_index = np.argmax(self.reward_estimates)
        return action_index
