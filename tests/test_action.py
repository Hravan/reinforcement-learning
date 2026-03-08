from reinforcement_learning.action import Action


def test_perform():
    action = Action(value=1, std=0)
    assert action.perform() == 1

def test_gaussian(mocker):
    mocker.patch('numpy.random.normal', return_value=2)
    action = Action.gaussian(0, 1)
    assert action.value == 2
    assert action.std == 1

def test_nonstationary(mocker):
    action = Action(0, 1, stationary=False)
    mocker.patch('numpy.random.normal', return_value=0.01)
    action.perform()
    assert action.value == 0.01
