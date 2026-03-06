from reinforcement_learning.agent import Agent
from reinforcement_learning.action import Action


def test_deterministic_action():
    action = Action(1, 0)
    agent = Agent(action)
    agent.act()
    assert agent.rewards == [1]
