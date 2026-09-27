import pytest

from reinforcement_learning.agent import (
    Agent,
    EpsilonGreedyAgent,
    ConstantStepSize,
    SampleAverageStepSize,
    IncrementalValueEstimation,
)
from reinforcement_learning.action import Action
from reinforcement_learning.environment import Environment


@pytest.fixture
def two_actions():
    optimal_action = Action(1, 0)
    suboptimal_action = Action(-1, 0)
    return [optimal_action, suboptimal_action]


def test_random_choice(two_actions, mocker):
    environment = Environment(*two_actions)
    agent = Agent()
    mocker.patch('random.randrange', return_value=1)
    agent.act(environment)
    assert agent.experience.action_history == [1]
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    assert agent.experience.action_history == [1, 0]


def test_deterministic_action_history():
    action = Action(1, 0)
    environment = Environment(action)
    agent = Agent()
    agent.act(environment)
    assert agent.reward_estimates == [1]
    agent.act(environment)
    assert agent.reward_estimates == [1]

def test_two_deterministic_actions_history(mocker):
    optimal_action = Action(1, 0)
    suboptimal_action = Action(-1, 0)
    environment = Environment(optimal_action, suboptimal_action)
    agent = Agent()
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    mocker.patch('random.randrange', return_value=1)
    agent.act(environment)

    assert agent.reward_estimates == [1, -1]

def test_epsilon_greedy_action_choice(mocker):
    optimal_action = Action(1, 0)
    subotpimal_action = Action(-1, 0)
    environment = Environment(optimal_action, subotpimal_action)
    agent = EpsilonGreedyAgent(epsilon=0.1)
    mocker.patch('random.random', return_value=0.91)
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    mocker.patch('random.randrange', return_value=1)
    agent.act(environment)
    mocker.patch('random.random', return_value=0.1)
    agent.act(environment)
    assert agent.experience.action_history == [0, 1, 0]


def test_mean_reward(mocker):
    optimal_action = Action(1, 0)
    subotpimal_action = Action(-1, 0)
    environment = Environment(optimal_action, subotpimal_action)
    agent = Agent()
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    mocker.patch('random.randrange', return_value=1)
    agent.act(environment)

    assert agent.mean_reward == 0


def test_step_size_constant(two_actions, mocker):
    environment = Environment(*two_actions)
    agent = Agent(value_estimation_method=IncrementalValueEstimation(2, step_size_method=ConstantStepSize(0.4)))
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    assert agent.reward_estimates[0] == 0.4


def test_custom_action_selection():
    agent = Agent(action_selection_method=lambda context: 1)
    assert agent.action_selection_method(agent._action_selection_context) == 1
    assert agent.action_selection_method(agent._action_selection_context) == 1
    assert agent.action_selection_method(agent._action_selection_context) == 1


def test_sample_average_step_size():
    step_size_method = SampleAverageStepSize()
    environment = Environment(Action(1, 0))
    agent = Agent()
    agent.act(environment)
    assert step_size_method(agent) == 1
    agent.act(environment)
    assert step_size_method(agent) == 0.5
    agent.act(environment)
    assert step_size_method(agent) == 1 / 3


def test_incremental_value_estimation_initial_value():
    value_estimation_method = IncrementalValueEstimation(3, initial_value=5)
    assert value_estimation_method.estimates == [5, 5, 5]


def test_incremental_value_estimation_default_step_size():
    value_estimation_method = IncrementalValueEstimation(1)
    assert isinstance(value_estimation_method.step_size_method, SampleAverageStepSize)


def test_incremental_value_estimation_update():
    value_estimation_method = IncrementalValueEstimation(2, step_size_method=ConstantStepSize(0.4))
    value_estimation_method.update(None, 0, 1)
    assert value_estimation_method.estimates == [0.4, 0]


def test_agent_infers_n_actions_from_first_environment(two_actions):
    environment = Environment(*two_actions)
    agent = Agent()
    assert agent.n_actions is None
    agent.act(environment)
    assert agent.n_actions == 2


def test_agent_default_value_estimation_method(two_actions):
    environment = Environment(*two_actions)
    agent = Agent()
    agent.act(environment)
    assert isinstance(agent.value_estimation_method, IncrementalValueEstimation)


def test_agent_without_value_estimation_method(mocker):
    action = Action(1, 0)
    environment = Environment(action)
    agent = Agent(value_estimation_method=None)
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    assert agent.value_estimation_method is None
    with pytest.raises(AttributeError):
        agent.reward_estimates
