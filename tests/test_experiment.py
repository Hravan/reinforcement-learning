import numpy as np

from reinforcement_learning.action import Action
from reinforcement_learning.agent import Agent
from reinforcement_learning.environment import Environment
from reinforcement_learning.action_selection import RandomActionSelection
from reinforcement_learning.experiment import Experiment, ExperimentResult, MeanReward, OptimalActionRate, RecentMeanReward


def test_mean_reward_metric():
    environment = Environment(Action(1, 0))
    agent = Agent(environment)
    agent.act(environment)
    metric = MeanReward()
    assert metric(agent, environment) == agent.mean_reward


def test_optimal_action_rate_metric(mocker):
    optimal = Action(1, 0)
    suboptimal = Action(-1, 0)
    environment = Environment(optimal, suboptimal)
    agent = Agent(environment)
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    metric = OptimalActionRate()
    assert metric(agent, environment) == True


def test_recent_mean_reward_metric(mocker):
    environment = Environment(Action(0, 1))
    agent = Agent(environment)
    mocker.patch('numpy.random.normal', side_effect=[1, 2, 3, 4])
    for _ in range(4):
        agent.act(environment)
    metric = RecentMeanReward(n=2)
    assert metric(agent, environment) == 3.5


def test_recent_mean_reward_metric_fewer_than_n_steps(mocker):
    environment = Environment(Action(0, 1))
    agent = Agent(environment)
    mocker.patch('numpy.random.normal', side_effect=[1, 2])
    agent.act(environment)
    agent.act(environment)
    metric = RecentMeanReward(n=10)
    assert metric(agent, environment) == 1.5


def test_metric_name_from_class_name():
    assert MeanReward().name == 'MeanReward'
    assert OptimalActionRate().name == 'OptimalActionRate'
    assert RecentMeanReward(n=5).name == 'RecentMeanReward'


def test_experiment_run_shape():
    experiment = Experiment(n_runs=2, n_steps=3, metric=MeanReward())
    agents = [lambda environment: Agent(environment), lambda environment: Agent(environment)]
    result = experiment.run(agents, lambda: [Action(1, 0)])
    assert result.raw.shape == (2, 2, 3)


def test_experiment_run_labels():
    experiment = Experiment(n_runs=1, n_steps=1, metric=MeanReward())
    agents = [lambda environment: Agent(environment, action_selection_method=RandomActionSelection())]
    result = experiment.run(agents, lambda: [Action(1, 0)])
    assert result.labels == [
        'Agent(action_selection_method=RandomActionSelection(), '
        'value_estimation_method=IncrementalValueEstimation(initial_value=0, step_size_method=SampleAverageStepSize()))'
    ]


def test_experiment_result_frame():
    raw = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    result = ExperimentResult(raw, labels=['agentA'], metric_name='mean_reward', group_name='agent')
    frame = result.frame
    assert list(frame.columns) == ['mean_reward', 'agent', 'step']
    assert frame['mean_reward'].tolist() == [2.0, 3.0]
    assert frame['step'].tolist() == [0, 1]
    assert frame['agent'].tolist() == ['agentA', 'agentA']


def test_experiment_result_plot():
    raw = np.array([[[1.0, 2.0]]])
    result = ExperimentResult(raw, labels=['agentA'], metric_name='mean_reward', group_name='agent')
    axes = result.plot()
    assert axes is not None
