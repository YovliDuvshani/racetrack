import random
from copy import deepcopy
from functools import cached_property
from typing import Tuple, Optional

from functools import lru_cache

import numpy as np

from config import (
    MAX_ACCELERATION,
    MAX_SPEED,
    SAMPLING_RATE_FOR_TRAJECTORY,
    NEGATIVE_REWARD,
)
from policy import Policy
from utils import State, Acceleration, Position, Speed, TimeStep


class Env:
    def __init__(self, map_path: str):
        self.map = self._load_map(map_path)

    @staticmethod
    def _load_map(map_path: str):
        with open(map_path) as f:
            map_txt = [row[:-1] for row in f.readlines()]

            map_array = np.zeros((len(map_txt), len(map_txt[0])))
            for i, row in enumerate(map_txt):
                for j, area in enumerate(row):
                    if area == "#":
                        map_array[i, j] = -1
                    elif area == "S":
                        map_array[i, j] = 0
                    elif area == "Z":
                        map_array[i, j] = 1
                    else:
                        map_array[i, j] = 2
        return map_array

    def generate_episode(self, policy: Policy):
        loop = True
        state = self._get_random_initial_state()
        episode = []
        while loop:
            action = policy.get_action_for_given_policy(state)
            current_state = deepcopy(state)
            state, reward, loop = self.transitions(state, action)
            if state and self.map[state.position] == 0: # Reset the episode each time the car is on S
                episode = []
            episode.append(TimeStep(current_state, action, reward))
        return episode

    def transitions(
        self, state: State, action: Acceleration
    ) -> Tuple[Optional[State], int, bool]:
        new_speed_x = state.speed.x
        new_speed_y = state.speed.y
        if abs(new_speed_x) < MAX_SPEED:
            new_speed_x += action.x
        if abs(new_speed_y) < MAX_SPEED:
            new_speed_y += action.y
        speed = Speed(new_speed_x, new_speed_y)

        new_position_candidate = Position(
            state.position.x + speed.x,
            state.position.y + speed.y,
        )
        ordered_path = self._get_ordered_path(state.position, new_position_candidate)
        next_state = State(new_position_candidate, speed)
        for area in ordered_path:
            if self.map[area] == 2:
                return None, NEGATIVE_REWARD, False
            elif self.map[area] == -1:
                next_state = self._get_random_initial_state()
                break

        return next_state, NEGATIVE_REWARD, True

    @lru_cache
    def _get_ordered_path(self, position: Position, next_position: Position):
        encountered_areas = []
        trajectory = [
            Position(round(i), round(j))
            for (i, j) in zip(
                np.linspace(position.x, next_position.x, SAMPLING_RATE_FOR_TRAJECTORY),
                np.linspace(position.y, next_position.y, SAMPLING_RATE_FOR_TRAJECTORY),
            )
        ]
        for encountered_area in trajectory:
            if encountered_area not in encountered_areas:
                encountered_areas.append(encountered_area)
        return encountered_areas

    def _get_random_initial_state(self):
        initial_position = Position(*random.choice(np.argwhere(self.map == 0)))
        return State(initial_position, Speed(0, 0))

    @cached_property
    def possible_actions(self):
        return [
            Acceleration(i, j)
            for i in range(-MAX_ACCELERATION, MAX_ACCELERATION + 1)
            for j in range(-MAX_ACCELERATION, MAX_ACCELERATION + 1)
            if abs(i) + abs(j) <= MAX_ACCELERATION
        ]
