from reinforcement_learning.action import Action


class TestAction:
    def test_perform(self):
        action = Action(value=1, std=0)
        assert action.perform() == 1