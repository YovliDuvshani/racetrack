from dataclasses import dataclass
from typing import Optional, NamedTuple

from config import MAX_ACCELERATION, MAX_SPEED


class Position(NamedTuple):
    x: int
    y: int


class Speed(NamedTuple):
    x: int
    y: int


class Acceleration(NamedTuple):
    x: int
    y: int


@dataclass
class State:
    position: Position
    speed: Speed


@dataclass
class TimeStep:
    state: Optional[State]
    action: Acceleration
    reward: int


ACTION_SHAPE = (2 * MAX_ACCELERATION + 1, 2 * MAX_ACCELERATION + 1)
SPEED_SHAPE = (2 * MAX_SPEED + 1, 2 * MAX_SPEED + 1)
