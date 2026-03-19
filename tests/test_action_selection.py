from reinforcement_learning.agent import Agent
from reinforcement_learning.action import Action
from reinforcement_learning.action_selection import UCB, GradientBandit

def test_ucb_one_action():
    action = Action.gaussian(mean=0, std=1)
    agent = Agent(action, action_selection_method=UCB(2))
    assert agent.select_action() == 0

def test_two_actions_first_exploit(mocker):
    action1 = Action.gaussian(mean=0, std=1)
    action2 = Action.gaussian(mean=0, std=1)
    agent = Agent(action1, action2, action_selection_method=UCB(2))
    mocker.patch('random.choice', return_value=0)
    agent.act()
    agent.act()
    agent.experience.action_history == [0, 1]


def test_explore_when_action_value_close_to_greedy():
    action1 = Action(10, 0)
    action2 = Action(9.5, 0)
    agent = Agent(action1, action2, action_selection_method=UCB(2))
    agent.act()
    agent.act()
    agent.act()
    # At this stage 1 is a nongreedy action with value close to the greedy action
    assert agent.select_action() == 1
    assert agent.reward_estimates[0] > agent.reward_estimates[1]


def test_gradient_bandit():
    gradient_bandit = GradientBandit(alpha=0.1, n_actions=2, baseline=0)
    action_index = 0
    reward = 1
    gradient_bandit.update_preferences(action_index, reward)
    assert gradient_bandit.action_preferences == [0.05, -0.05]


def test_select_with_gradient_bandit(mocker):
    action1 = Action.gaussian(0, 1)
    action2 = Action.gaussian(0, 1)
    gradient_bandit = GradientBandit(alpha=0.1, baseline=0)
    agent = Agent(action1, action2, action_selection_method=gradient_bandit)
    mocker.patch('random.choice', return_value=1)
    agent.act()
    gradient_bandit.action_preferences[1] == 0.1 * 0.5
    gradient_bandit.action_preferences[0] == -0.1 * 0.5
