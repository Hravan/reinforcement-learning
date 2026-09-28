import numpy as np


class Environment:
    """An environment an agent interacts with by choosing action indices.

    Args:
        *actions: The `Action`s available in this environment.
    """

    def __init__(self, *actions):
        self.actions = actions
        self.n_actions = len(actions)

    def step(self, action_index):
        """Performs the chosen action and advances the environment by one step.

        Args:
            action_index: Index of the action to perform.

        Returns:
            The reward received for that action.
        """
        reward = self.actions[action_index].perform()
        for action in self.actions:
            action.drift()
        return reward

    @property
    def optimal_action(self):
        """Index of the action with the highest true value."""
        return np.argmax([action.value for action in self.actions])
