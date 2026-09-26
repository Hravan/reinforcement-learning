from __future__ import annotations
import random

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from reinforcement_learning.agent import Agent


class ActionSelectionContext:
    """A narrow, explicit view of agent state for action selection.

    Args:
        agent: The `Agent` to build the context from.
    """

    def __init__(self, agent: Agent):
        self.n_actions = len(agent.actions)
        self.experience = agent.experience
        self.reward_estimates = (
            agent.reward_estimates if agent.value_estimation_method is not None else None
        )


class UCB:
    """The upper-confidence-bound (UCB) action-selection method.

    Args:
        exploration_coefficient: Controls the degree of exploration; a
            higher value favors less-explored actions more strongly.
    """

    def __init__(self, exploration_coefficient: float):
        self.exploration_coefficient = exploration_coefficient

    def __call__(self, context: ActionSelectionContext):
        """Chooses the action with the highest upper confidence bound.

        Args:
            context: The agent's `ActionSelectionContext`.

        Returns:
            Index of the chosen action.
        """
        criterion_values = []
        for action_index in range(context.n_actions):
            n_selected = context.experience.n_selected(action_index)

            if n_selected == 0:
                criterion_values.append(float('inf'))
                continue

            current_estimate = context.reward_estimates[action_index]
            timestep = len(context.experience)
            criterion_value = current_estimate + self.exploration_coefficient * (np.log(timestep) / n_selected) ** (1/2)
            criterion_values.append(criterion_value)
        return np.argmax(criterion_values)

    def update(self, *args, **kwargs):
        """No-op: UCB has no internal state to update from feedback."""
        pass


class GradientBandit:
    """The gradient-bandit action-selection method.

    Maintains action preferences and selects actions by sampling from the
    softmax distribution over them.

    Args:
        alpha: Step size for the preference update.
        baseline: Reward baseline the update compares against.
        n_actions: Number of actions, if known upfront. If omitted, the
            preferences are initialized lazily on the first call.
    """

    def __init__(self, alpha: float, baseline: float, n_actions: int | None = None):
        self.alpha = alpha
        self.baseline = baseline
        self.action_preferences = [0.0 for _ in range(n_actions)] if n_actions is not None else None

    def __call__(self, context: ActionSelectionContext):
        """Samples an action from the softmax distribution over preferences.

        Args:
            context: The agent's `ActionSelectionContext`.

        Returns:
            Index of the chosen action.
        """
        if self.action_preferences is None:
            self.action_preferences = [0.0 for _ in range(context.n_actions)]
        action_choice = random.choices(range(context.n_actions), weights=self.action_probabilities, k=1)
        return action_choice[0]

    @property
    def action_probabilities(self):
        """Softmax probabilities over the current action preferences."""
        preferences_exponentiated = [np.exp(preference) for preference in self.action_preferences]
        denominator = sum(preferences_exponentiated)
        probabilities = [preference / denominator for preference in preferences_exponentiated]
        return probabilities

    def update(self, action_index: int, reward: float):
        """Updates every action's preference towards/away from the reward.

        Args:
            action_index: Index of the action that was chosen.
            reward: The reward received for that action.
        """
        probabilities = self.action_probabilities
        self.action_preferences[action_index] += self.alpha * (reward - self.baseline) * (1 - probabilities[action_index])
        for i, _ in enumerate(self.action_preferences):
            if i != action_index:
                self.action_preferences[i] += -self.alpha * (reward - self.baseline) * probabilities[i]


class RandomActionSelection:
    """An action-selection method that picks uniformly at random."""

    def __call__(self, context: ActionSelectionContext):
        """Chooses an action index uniformly at random.

        Args:
            context: The agent's `ActionSelectionContext`.

        Returns:
            Index of the chosen action.
        """
        return random.randrange(context.n_actions)

    def update(self, *args, **kwargs):
        """No-op: random selection has no internal state to update."""
        pass


class EpsilonGreedy:
    """The epsilon-greedy action-selection method.

    Args:
        epsilon: Probability of choosing a random action instead of the
            greedy one.
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, context: ActionSelectionContext):
        """Chooses a random action with probability epsilon, else the greedy one.

        Args:
            context: The agent's `ActionSelectionContext`.

        Returns:
            Index of the chosen action.
        """
        if random.random() > 1 - self.epsilon:
            action_index = random.randrange(context.n_actions)
        else:
            action_index = np.argmax(context.reward_estimates)
        return action_index

    def update(self, *args, **kwargs):
        """No-op: epsilon-greedy has no internal state to update."""
        pass
