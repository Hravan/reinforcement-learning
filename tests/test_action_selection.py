from reinforcement_learning.agent import Agent
from reinforcement_learning.action import Action
from reinforcement_learning.environment import Environment
from reinforcement_learning.action_selection import UCB, GradientBandit

def test_action_selection_context_with_value_estimation(mocker):
    action1 = Action(1, 0)
    action2 = Action(2, 0)
    environment = Environment(action1, action2)
    agent = Agent(environment)
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    context = agent._action_selection_context
    assert context.n_actions == 2
    assert context.experience is agent.experience
    assert context.reward_estimates == agent.reward_estimates


def test_action_selection_context_without_value_estimation(mocker):
    action1 = Action(1, 0)
    action2 = Action(2, 0)
    environment = Environment(action1, action2)
    agent = Agent(environment, value_estimation_method=None)
    mocker.patch('random.randrange', return_value=0)
    agent.act(environment)
    assert agent._action_selection_context.reward_estimates is None


def test_ucb_one_action():
    action = Action.gaussian(mean=0, std=1)
    environment = Environment(action)
    agent = Agent(environment, action_selection_method=UCB(2))
    agent.act(environment)
    assert agent.experience.action_history == [0]

def test_two_actions_first_exploit(mocker):
    action1 = Action.gaussian(mean=0, std=1)
    action2 = Action.gaussian(mean=0, std=1)
    environment = Environment(action1, action2)
    agent = Agent(environment, action_selection_method=UCB(2))
    mocker.patch('random.choice', return_value=0)
    agent.act(environment)
    agent.act(environment)
    agent.experience.action_history == [0, 1]


def test_explore_when_action_value_close_to_greedy():
    action1 = Action(10, 0)
    action2 = Action(9.5, 0)
    environment = Environment(action1, action2)
    agent = Agent(environment, action_selection_method=UCB(2))
    agent.act(environment)
    agent.act(environment)
    agent.act(environment)
    # At this stage 1 is a nongreedy action with value close to the greedy action
    assert agent.action_selection_method(agent._action_selection_context) == 1
    assert agent.reward_estimates[0] > agent.reward_estimates[1]


def test_gradient_bandit():
    gradient_bandit = GradientBandit(alpha=0.1, n_actions=2, baseline=0)
    action_index = 0
    reward = 1
    gradient_bandit.update(action_index, reward)
    assert gradient_bandit.action_preferences == [0.05, -0.05]


def test_select_with_gradient_bandit(mocker):
    action1 = Action.gaussian(1, 0)
    action2 = Action.gaussian(1, 0)
    environment = Environment(action1, action2)
    gradient_bandit = GradientBandit(alpha=0.1, baseline=0)
    agent = Agent(environment, action_selection_method=gradient_bandit, value_estimation_method=None)
    mocker.patch('random.choices', return_value=[1])
    agent.act(environment)
    assert gradient_bandit.action_preferences[1] == 0.1 * 0.5
    assert gradient_bandit.action_preferences[0] == -0.1 * 0.5
