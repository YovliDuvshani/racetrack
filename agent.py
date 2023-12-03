from copy import deepcopy

import numpy as np

from config import GAMMA, NB_ROUNDS, MAX_SPEED, MAX_ACCELERATION, EPSILON_MIN
from env import Env
from policy import Policy
from policy_manager import PolicyManager
from utils import SPEED_SHAPE, ACTION_SHAPE, State, Position, Speed


class Agent:
    def __init__(self, env: Env, policy_manager: PolicyManager):
        self.env = env
        self.policy_manager = policy_manager
        self.q = (
            np.zeros(
                (
                    *env.map.shape,
                    *SPEED_SHAPE,
                    *ACTION_SHAPE,
                )
            )
            - 1000
        )
        for action in self.env.possible_actions:
            self.q[
                :, :, :, :, action[0] + MAX_ACCELERATION, action[1] + MAX_ACCELERATION
            ] = -30
        self.weights = np.zeros_like(self.q)
        self.b = policy_manager.random_policy
        self.policy = Policy(self.b.policy.copy())

    def monte_carlo_control(self):
        for i in range(NB_ROUNDS):
            episode = self.env.generate_episode(self.b)
            G = 0
            W = 1
            for j, t_s in enumerate(episode[::-1][:-1]):
                G = GAMMA * G + t_s.reward
                self.weights[
                    t_s.state.position.x,
                    t_s.state.position.y,
                    t_s.state.speed.x + MAX_SPEED,
                    t_s.state.speed.y + MAX_SPEED,
                    t_s.action.x + MAX_ACCELERATION,
                    t_s.action.y + MAX_ACCELERATION,
                ] += W
                self.q[
                    t_s.state.position.x,
                    t_s.state.position.y,
                    t_s.state.speed.x + MAX_SPEED,
                    t_s.state.speed.y + MAX_SPEED,
                    t_s.action.x + MAX_ACCELERATION,
                    t_s.action.y + MAX_ACCELERATION,
                ] += (
                    W
                    / self.weights[
                        t_s.state.position.x,
                        t_s.state.position.y,
                        t_s.state.speed.x + MAX_SPEED,
                        t_s.state.speed.y + MAX_SPEED,
                        t_s.action.x + MAX_ACCELERATION,
                        t_s.action.y + MAX_ACCELERATION,
                    ]
                    * (
                        G
                        - self.q[
                            t_s.state.position.x,
                            t_s.state.position.y,
                            t_s.state.speed.x + MAX_SPEED,
                            t_s.state.speed.y + MAX_SPEED,
                            t_s.action.x + MAX_ACCELERATION,
                            t_s.action.y + MAX_ACCELERATION,
                        ]
                    )
                )
                self.policy.policy[
                    t_s.state.position.x,
                    t_s.state.position.y,
                    t_s.state.speed.x + MAX_SPEED,
                    t_s.state.speed.y + MAX_SPEED,
                ] = np.zeros(ACTION_SHAPE)
                self.policy.policy[
                    t_s.state.position.x,
                    t_s.state.position.y,
                    t_s.state.speed.x + MAX_SPEED,
                    t_s.state.speed.y + MAX_SPEED,
                    self._get_action_with_highest_action_state_value(t_s.state)[0],
                    self._get_action_with_highest_action_state_value(t_s.state)[1],
                ] = 1
                if t_s.action != self.policy.get_action_for_given_policy(
                    t_s.state
                ):  # Assumption made that the target policy is greedy
                    print(f"Iteration {i} was interrupted - {j+1} action-states modified")
                    break
                W /= self.b.policy[
                    t_s.state.position.x,
                    t_s.state.position.y,
                    t_s.state.speed.x + MAX_SPEED,
                    t_s.state.speed.y + MAX_SPEED,
                    t_s.action.x + MAX_ACCELERATION,
                    t_s.action.y + MAX_ACCELERATION,
                ]
            self.b = self.policy_manager.transform_policy_to_epsilon_greedy(
                deepcopy(self.policy), max(1 - i / NB_ROUNDS, EPSILON_MIN)
            )

    def _get_action_with_highest_action_state_value(self, state: State):
        idx_max = (
            self.q[
                state.position.x,
                state.position.y,
                state.speed.x + MAX_SPEED,
                state.speed.y + MAX_SPEED,
            ]
            .reshape(np.prod(ACTION_SHAPE))
            .argmax(axis=-1)
        )
        return np.stack(np.unravel_index(idx_max, ACTION_SHAPE))

    def v(self):
        v = np.zeros((*self.env.map.shape, *SPEED_SHAPE))
        for i in range(self.env.map.shape[0]):
            for j in range(self.env.map.shape[1]):
                for k in range(SPEED_SHAPE[0]):
                    for l in range(SPEED_SHAPE[1]):
                        action_max = self._get_action_with_highest_action_state_value(
                            State(Position(i, j), Speed(k - MAX_SPEED, l - MAX_SPEED))
                        )
                        v[i, j, k, l] = self.q[i, j, k, l, action_max[0], action_max[1]]
        return v
