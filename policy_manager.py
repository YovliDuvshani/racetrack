from functools import cached_property

import numpy as np

from config import MAX_ACCELERATION
from env import Env
from policy import Policy
from utils import ACTION_SHAPE, SPEED_SHAPE


class PolicyManager:
    def __init__(self, env: Env):
        self.env = env

    @cached_property
    def random_policy(self) -> Policy:
        random_policy = np.zeros(
            (
                *self.env.map.shape,
                *SPEED_SHAPE,
                *ACTION_SHAPE,
            )
        )
        for action in self.env.possible_actions:
            random_policy[
                :, :, :, :, action.x + MAX_ACCELERATION, action.y + MAX_ACCELERATION
            ] = 1 / len(self.env.possible_actions)
        return Policy(random_policy)

    def transform_policy_to_epsilon_greedy(self, policy: Policy, epsilon: float):
        for action in self.env.possible_actions:
            policy.policy[
                :, :, :, :, action.x + MAX_ACCELERATION, action.y + MAX_ACCELERATION
            ] *= (1 - epsilon)
            policy.policy[
                :, :, :, :, action.x + MAX_ACCELERATION, action.y + MAX_ACCELERATION
            ] += epsilon / len(self.env.possible_actions)
        return policy
