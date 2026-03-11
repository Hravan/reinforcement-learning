import pytest

from reinforcement_learning.agent import Agent, EpsilonGreedyAgent
from reinforcement_learning.action import Action


@pytest.fixture
def two_actions():
    optimal_action = Action(1, 0)
    subotpimal_action = Action(-1, 0)
    return [optimal_action, subotpimal_action]


def test_deterministic_action_history():
    action = Action(1, 0)
    agent = Agent(action)
    agent.act()
    assert agent.reward_estimates == [1]
    agent.act()
    assert agent.reward_estimates == [1]

def test_two_deterministic_actions_history(mocker):
    optimal_action = Action(1, 0)
    subotpimal_action = Action(-1, 0)
    agent = Agent(optimal_action, subotpimal_action)
    mocker.patch('random.randrange', return_value=0)
    agent.act()
    mocker.patch('random.randrange', return_value=1)
    agent.act()
    
    assert agent.reward_estimates == [1, -1]

def test_epsilon_greedy_action_choice(mocker):
    optimal_action = Action(1, 0)
    subotpimal_action = Action(-1, 0)
    agent = EpsilonGreedyAgent(optimal_action, subotpimal_action, epsilon=0.1)
    mocker.patch('random.random', return_value=0.91)
    mocker.patch('random.randrange', return_value=0)
    agent.act()
    mocker.patch('random.randrange', return_value=1)
    agent.act()
    mocker.patch('random.random', return_value=0.1)
    agent.act()
    assert agent.action_history == [0, 1, 0]


def test_optimal_action():
    optimal_action = Action(1, 0)
    subotpimal_action = Action(-1, 0)
    agent = Agent(optimal_action, subotpimal_action)
    assert agent.optimal_action == 0


def test_mean_reward(mocker):
    optimal_action = Action(1, 0)
    subotpimal_action = Action(-1, 0)
    agent = Agent(optimal_action, subotpimal_action)
    mocker.patch('random.randrange', return_value=0)
    agent.act()
    mocker.patch('random.randrange', return_value=1)
    agent.act()
    
    assert agent.mean_reward == 0


def test_step_size_constant(two_actions, mocker):
    agent = Agent(*two_actions, step_size=lambda ctx: 0.4)
    mocker.patch('random.randrange', return_value=0)
    agent.act()
    assert agent.reward_estimates[0] == 0.4
