from reinforcement_learning.agent import Agent
from reinforcement_learning.action import Action
from reinforcement_learning.action_selection import UCB

def test_ucb_one_action():
    action = Action.gaussian(mean=0, std=1)
    agent = Agent(action, action_selection_method=UCB(2))
    assert agent.select_action() == 0

def test_two_actions_first_exploin(mocker):
    action1 = Action.gaussian(mean=0, std=1)
    action2 = Action.gaussian(mean=0, std=1)
    agent = Agent(action1, action2, action_selection_method=UCB(2))
    mocker.patch('random.choice', return_value=0)
    agent.act()
    agent.act()
    agent.action_history == [0, 1]


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
