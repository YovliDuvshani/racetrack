import numpy as np

from config import MAX_ACCELERATION, MAX_SPEED
from utils import State, Acceleration, ACTION_SHAPE


class Policy:
    def __init__(self, policy: np.array):
        self.policy = policy

    def get_action_for_given_policy(self, state: State):
        action_probabilities = self.policy[
            state.position.x,
            state.position.y,
            state.speed.x + MAX_SPEED,
            state.speed.y + MAX_SPEED,
            :,
            :,
        ].reshape(np.prod(ACTION_SHAPE))
        indented_action = np.unravel_index(
            np.random.choice(
                list(range(len(action_probabilities))), p=action_probabilities
            ),
            ACTION_SHAPE,
        )
        return Acceleration(
            indented_action[0] - MAX_ACCELERATION,
            indented_action[1] - MAX_ACCELERATION,
        )
