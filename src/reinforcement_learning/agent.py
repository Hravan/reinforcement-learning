import random
import numpy as np


class StepSizeContext:
    def __init__(self, action_history, action_index):
        self.action_history = action_history
        self.action_index = action_index
    
    @property
    def n_choices(self):
        return sum(1 for action in self.action_history if action == self.action_index)


class Agent:
    def __init__(self, *actions, initial_reward_value=0, step_size=lambda ctx: 1 / ctx.n_choices):
        self.actions = actions
        self.reward_estimates = [initial_reward_value for _ in self.actions]
        self.action_history = []
        self.sum_rewards = 0
        self.step_size = step_size
    
    def act(self):
        action_index = self.choice()
        self.action_history.append(action_index)
        reward = self.actions[action_index].perform()
        self.sum_rewards += reward

        current_estimate = self.reward_estimates[action_index]
        ctx = StepSizeContext(self.action_history, action_index)
        self.reward_estimates[action_index] = current_estimate + self.step_size(ctx) * (reward - current_estimate)
    
    def choice(self):
        action_index = random.randrange(len(self.actions))
        return action_index
    
    @property
    def optimal_action(self):
        return np.argmax([action.value for action in self.actions])
    
    @property
    def mean_reward(self):
        return self.sum_rewards / len(self.action_history)
    

class EpsilonGreedyAgent(Agent):
    def __init__(self, *actions, epsilon=0, **kwargs):
        super().__init__(*actions, **kwargs)
        self.epsilon = epsilon

    def choice(self):
        if random.random() > 1 - self.epsilon:
            action_index = random.randrange(len(self.actions))
        else:
            action_index = np.argmax(self.reward_estimates)
        return action_index
