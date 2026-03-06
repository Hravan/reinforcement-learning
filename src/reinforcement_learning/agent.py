import random


class Agent:
    def __init__(self, *actions):
        self.actions = actions
        self.rewards = [[] for _ in self.actions]
    
    def act(self):
        action_index = random.randrange(len(self.actions))
        reward = self.actions[action_index].perform()
        self.rewards[action_index].append(reward)
