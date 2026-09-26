from reinforcement_learning.action import Action, Experience


def test_perform():
    action = Action(value=1, std=0)
    assert action.perform() == 1

def test_gaussian(mocker):
    mocker.patch('numpy.random.normal', return_value=2)
    action = Action.gaussian(0, 1)
    assert action.value == 2
    assert action.std == 1

def test_perform_does_not_drift(mocker):
    action = Action(0, 1, stationary=False)
    mocker.patch('numpy.random.normal', return_value=0.01)
    action.perform()
    assert action.value == 0

def test_drift(mocker):
    action = Action(0, 1, stationary=False)
    mocker.patch('numpy.random.normal', return_value=0.01)
    action.drift()
    assert action.value == 0.01

def test_drift_stationary_is_noop(mocker):
    action = Action(0, 1, stationary=True)
    mocker.patch('numpy.random.normal', return_value=0.01)
    action.drift()
    assert action.value == 0

def test_repr():
    action = Action(0, 1)
    assert repr(action) == 'Action(value=0, std=1, stationary=True)'


def test_experience_n_selected():
    experience = Experience()
    experience.update(0, 1)
    experience.update(1, 2)
    experience.update(0, 3)
    assert experience.n_selected(0) == 2
    assert experience.n_selected(1) == 1
    assert experience.n_selected(2) == 0


def test_experience_last_action_and_n_selected_last_action():
    experience = Experience()
    experience.update(0, 1)
    experience.update(1, 2)
    assert experience.last_action == 1
    assert experience.n_selected_last_action() == 1
    experience.update(1, 3)
    assert experience.n_selected_last_action() == 2


def test_experience_mean_reward():
    experience = Experience()
    experience.update(0, 1)
    experience.update(1, 3)
    assert experience.mean_reward == 2


def test_experience_len():
    experience = Experience()
    experience.update(0, 1)
    experience.update(1, 2)
    assert len(experience) == 2
