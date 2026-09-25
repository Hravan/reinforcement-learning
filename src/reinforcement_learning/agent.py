import numpy as np

from reinforcement_learning.action import Experience
from reinforcement_learning.action_selection import RandomActionSelection, EpsilonGreedy


_DEFAULT_VALUE_ESTIMATION_METHOD = object()


class Agent:
    """A bandit agent that selects actions and learns reward estimates.

    Composes an action-selection method (which action to choose next) and a
    value-estimation method (how to learn per-action reward estimates from
    experience) to implement one of the action-value methods from chapter 2.
    """

    def __init__(self, *actions, action_selection_method=None, value_estimation_method=_DEFAULT_VALUE_ESTIMATION_METHOD):
        """Initializes the agent.

        Args:
            *actions: The `Action`s the agent can choose between.
            action_selection_method: Callable `(agent) -> int` deciding
                which action to choose next. Defaults to
                `RandomActionSelection()`.
            value_estimation_method: `ValueEstimationMethod` maintaining
                per-action reward estimates. Defaults to
                `IncrementalValueEstimation(len(actions))`. Pass `None`
                explicitly for a method that needs no value estimates (e.g.
                a `GradientBandit`-driven agent).
        """
        self.actions = actions
        self.experience = Experience()
        self.action_selection_method = (
            action_selection_method if action_selection_method is not None else RandomActionSelection()
        )
        if value_estimation_method is _DEFAULT_VALUE_ESTIMATION_METHOD:
            value_estimation_method = IncrementalValueEstimation(len(actions))
        self.value_estimation_method = value_estimation_method

    def act(self):
        """Chooses an action, performs it, and updates the agent's state."""
        action_index = self.action_selection_method(self)
        reward = self.actions[action_index].perform()
        self.experience.update(action_index, reward)
        self.action_selection_method.update(action_index, reward)
        if self.value_estimation_method is not None:
            self.value_estimation_method.update(self, action_index, reward)

    @property
    def optimal_action(self):
        """Index of the action with the highest true value."""
        return np.argmax([action.value for action in self.actions])

    @property
    def mean_reward(self):
        """Mean reward obtained by the agent so far."""
        return self.experience.mean_reward

    @property
    def reward_estimates(self):
        """Per-action reward estimates, from `value_estimation_method`."""
        return self.value_estimation_method.estimates

    def n_selected(self, action_index):
        """Counts how many times an action has been chosen so far.

        Args:
            action_index: Index of the action into `self.actions`.

        Returns:
            The number of times this action appears in the history.
        """
        return self.experience.n_selected(action_index)

    @property
    def n_choices(self):
        '''Number of choices already made by the agent.'''
        return len(self.experience)


class EpsilonGreedyAgent(Agent):
    """An agent that selects actions using the epsilon-greedy method."""

    def __init__(self, *actions, epsilon=0, **kwargs):
        """Initializes the agent.

        Args:
            *actions: The `Action`s the agent can choose between.
            epsilon: Probability of choosing a random action instead of the
                greedy one.
            **kwargs: Forwarded to `Agent.__init__`.
        """
        super().__init__(*actions, action_selection_method=EpsilonGreedy(epsilon), **kwargs)


class ConstantStepSize:
    """A step-size method that always returns the same, constant value."""

    def __init__(self, step_size):
        """Initializes the step-size method.

        Args:
            step_size: The constant step size to always return.
        """
        self.step_size = step_size

    def __call__(self, agent):
        """Returns the constant step size.

        Args:
            agent: The `Agent` requesting a step size (unused).

        Returns:
            The constant step size.
        """
        return self.step_size


class SampleAverageStepSize:
    """A step-size method that computes the running sample average."""

    def __call__(self, agent):
        """Returns 1 divided by the number of times the last action was chosen.

        Args:
            agent: The `Agent` requesting a step size.

        Returns:
            The sample-average step size for the agent's most recently
            chosen action.
        """
        return 1 / agent.experience.n_selected_last_action()


class IncrementalValueEstimation:
    """A value-estimation method that incrementally updates reward estimates."""

    def __init__(self, n_actions, initial_value=0, step_size_method=None):
        """Initializes the value-estimation method.

        Args:
            n_actions: Number of actions to keep a reward estimate for.
            initial_value: Initial value for every action's reward estimate.
            step_size_method: Callable `(agent) -> float` deciding how much
                weight to give the most recent reward. Defaults to
                `SampleAverageStepSize()`.
        """
        self.estimates = [initial_value for _ in range(n_actions)]
        self.step_size_method = step_size_method if step_size_method is not None else SampleAverageStepSize()

    def update(self, agent, action_index, reward):
        """Updates the estimate for the chosen action towards the reward.

        Args:
            agent: The `Agent` this method belongs to, passed on to
                `step_size_method`.
            action_index: Index of the action that was chosen.
            reward: The reward received for that action.
        """
        current_estimate = self.estimates[action_index]
        self.estimates[action_index] = current_estimate + self.step_size_method(agent) * (reward - current_estimate)
