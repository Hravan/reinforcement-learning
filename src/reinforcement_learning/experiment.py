import copy
from functools import cached_property

import numpy as np
import pandas as pd
import seaborn as sns

from reinforcement_learning.environment import Environment


class Metric:
    """A metric computed from an agent and environment after each step."""

    @property
    def name(self):
        """The metric's display name, derived from its class name."""
        return type(self).__name__

    def __call__(self, agent, environment):
        raise NotImplementedError


class MeanReward(Metric):
    """An agent's mean reward so far."""

    def __call__(self, agent, environment):
        return agent.mean_reward


class OptimalActionRate(Metric):
    """Whether the chosen action was the environment's optimal one."""

    def __call__(self, agent, environment):
        return agent.experience.last_action == environment.optimal_action


class Experiment:
    """Runs and scores agents against environments, repeated over many runs.

    Args:
        n_runs: Number of independent runs to average over.
        n_steps: Number of steps per run.
        metric: The `Metric` computed after each step.
    """

    def __init__(self, n_runs, n_steps, metric):
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.metric = metric

    def run(self, agents, action_factory, group_name='agent'):
        """Runs every agent against its own environment, repeated over many runs.

        Args:
            agents: A list of callables `(environment) -> Agent`.
            action_factory: Callable `() -> list[Action]` producing a fresh
                set of actions for each run.
            group_name: Column name for the agent label in the result's
                aggregated frame.

        Returns:
            The `ExperimentResult`.
        """
        labels = [None] * len(agents)
        raw = np.zeros((len(agents), self.n_runs, self.n_steps))
        for irun in range(self.n_runs):
            actions = action_factory()
            for iagent, agent_factory in enumerate(agents):
                environment = Environment(*copy.deepcopy(actions))
                agent = agent_factory(environment)
                if irun == 0:
                    labels[iagent] = repr(agent)
                for istep in range(self.n_steps):
                    agent.act(environment)
                    raw[iagent, irun, istep] = self.metric(agent, environment)
        return ExperimentResult(raw, labels, self.metric.name, group_name)


class ExperimentResult:
    """The outcome of an `Experiment` run.

    Args:
        raw: Array of shape `(n_agents, n_runs, n_steps)`.
        labels: Per-agent labels, length `n_agents`.
        metric_name: Column name for the metric in `frame`.
        group_name: Column name for the agent label in `frame`.
    """

    def __init__(self, raw, labels, metric_name, group_name):
        self.raw = raw
        self.labels = labels
        self.metric_name = metric_name
        self.group_name = group_name

    @cached_property
    def frame(self):
        """A long-format `DataFrame`, averaged over runs: one row per (agent, step)."""
        mean_per_agent = self.raw.mean(axis=1)
        n_agents, n_steps = mean_per_agent.shape
        return pd.DataFrame({
            self.metric_name: mean_per_agent.reshape(-1),
            self.group_name: pd.Categorical([label for label in self.labels for _ in range(n_steps)]),
            'step': [s for _ in range(n_agents) for s in range(n_steps)],
        })

    def plot(self):
        """Plots `frame` as a line plot of the metric over steps, grouped by agent."""
        return sns.lineplot(self.frame, x='step', y=self.metric_name, hue=self.group_name)
