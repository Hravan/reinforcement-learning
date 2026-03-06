from reinforcement_learning.action import Action


class TestAction:
    def test_perform(self):
        action = Action(value=1, std=0)
        assert action.perform() == 1
    
    def test_gaussian(self, mocker):
        mocker.patch('numpy.random.normal', return_value=2)
        action = Action.gaussian(0, 1)
        assert action.value == 2
        assert action.std == 1
