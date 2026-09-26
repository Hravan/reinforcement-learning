from typing import TypeAlias
import numpy as np

Reward: TypeAlias = float

class Action:
    """A bandit action drawn from a Gaussian reward distribution."""

    def __init__(self, value: float, std: float, stationary=True):
        """Initializes the action.

        Args:
            value: The action's true value (mean reward).
            std: Standard deviation of the reward distribution.
            stationary: If False, `drift()` takes a small random walk step
                each time it is called.
        """
        self.value = value
        self.std = std
        self.stationary = stationary

    def perform(self) -> Reward:
        """Samples a reward from the action's current distribution.

        Returns:
            The sampled reward.
        """
        return np.random.normal(self.value, self.std)

    def drift(self):
        """Applies one random-walk step to the true value, if nonstationary.

        No-op for a stationary action.
        """
        if not self.stationary:
            self.value = self.value + np.random.normal(0, 0.01)

    @classmethod
    def gaussian(cls, mean, std, **kwargs):
        """Creates an action whose true value is itself sampled from N(mean, std).

        Args:
            mean: Mean of the distribution the true value is sampled from.
            std: Standard deviation of both that distribution and the
                action's own reward distribution.
            **kwargs: Forwarded to `Action.__init__`.

        Returns:
            The new `Action`.
        """
        value = np.random.normal(mean, std)
        return cls(value, std, **kwargs)

    def __repr__(self):
        return f'Action(value={self.value}, std={self.std}, stationary={self.stationary})'


class Experience:
    """The history of actions chosen and rewards received by an agent."""

    def __init__(self):
        """Initializes an empty experience."""
        self.action_history = []
        self.sum_rewards = 0
        self._n_selected = {}

    def update(self, action, reward):
        """Records a chosen action and the reward it produced.

        Args:
            action: Index of the action that was chosen.
            reward: The reward received for that action.
        """
        self.action_history.append(action)
        self.sum_rewards += reward
        self._n_selected[action] = self._n_selected.get(action, 0) + 1

    def n_selected(self, action_index):
        """Counts how many times an action has been chosen so far.

        Args:
            action_index: Index of the action into the agent's actions.

        Returns:
            The number of times this action appears in the history.
        """
        return self._n_selected.get(action_index, 0)

    @property
    def last_action(self):
        """Index of the most recently chosen action."""
        return self.action_history[-1]

    def n_selected_last_action(self):
        """Counts how many times the most recently chosen action has been chosen."""
        return self.n_selected(self.last_action)

    def __len__(self):
        """Number of actions chosen so far."""
        return len(self.action_history)

    @property
    def mean_reward(self):
        """Mean reward received so far."""
        return self.sum_rewards / len(self.action_history)
