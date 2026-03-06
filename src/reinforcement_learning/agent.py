import random


class Agent:
    def __init__(self, *actions):
        self.actions = actions
        self.rewards = []
    
    def act(self):
        reward = random.choice(self.actions).perform()
        self.rewards.append(reward)
