import random
import numpy as np


class Agent:
    def __init__(self, *actions):
        self.actions = actions
        self.rewards = [[] for _ in self.actions]
        self.action_history = []
    
    def act(self):
        action_index = self.choice()
        self.action_history.append(action_index)
        reward = self.actions[action_index].perform()
        self.rewards[action_index].append(reward)
    
    def choice(self):
        action_index = random.randrange(len(self.actions))
        return action_index
    
    @property
    def optimal_action(self):
        return np.argmax(action.value for action in self.actions)


class EpsilonGreedyAgent(Agent):
    def __init__(self, *actions, epsilon=0):
        super().__init__(*actions)
        self.epsilon = epsilon

    def choice(self):
        if random.random() > 1 - self.epsilon:
            action_index = random.randrange(len(self.actions))
        else:
            reward_means = [np.mean(action_rewards) for action_rewards in self.rewards]
            action_index = np.argmax(reward_means)
        return action_index
