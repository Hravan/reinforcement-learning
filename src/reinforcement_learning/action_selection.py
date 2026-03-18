import numpy as np

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
