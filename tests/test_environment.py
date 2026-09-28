from reinforcement_learning.environment import Environment
from reinforcement_learning.action import Action


def test_n_actions():
    environment = Environment(Action(1, 0), Action(-1, 0))
    assert environment.n_actions == 2


def test_step_samples_reward():
    action = Action(value=1, std=0)
    environment = Environment(action)
    assert environment.step(0) == 1


def test_step_drifts_every_action_not_just_the_chosen_one(mocker):
    action0 = Action(0, 1, stationary=False)
    action1 = Action(0, 1, stationary=False)
    environment = Environment(action0, action1)
    mocker.patch('numpy.random.normal', return_value=0.01)
    environment.step(0)
    assert action0.value == 0.01
    assert action1.value == 0.01


def test_optimal_action():
    optimal_action = Action(1, 0)
    suboptimal_action = Action(-1, 0)
    environment = Environment(optimal_action, suboptimal_action)
    assert environment.optimal_action == 0
    environment = Environment(suboptimal_action, optimal_action)
    assert environment.optimal_action == 1
