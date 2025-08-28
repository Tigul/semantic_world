from __future__ import annotations

import abc
from dataclasses import field, dataclass
from itertools import chain
from typing import Iterable, Tuple, Set

import numpy as np

from .connections import ActiveConnection
from .world import World
from .world_entity import Body


def is_body_controlled(body: Body) -> bool:
    world = body._world
    root_part, tip_part = world.compute_split_chain_of_connections(world.root, body)
    connections = root_part + tip_part
    for c in connections:
        if (isinstance(c, ActiveConnection)
                and c.is_controlled
                and not c.frozen_for_collision_avoidance):
            return True
    return False

def sort_bodies(body_a: Body, body_b: Body) -> Tuple[Body, Body]:
    if body_a.name > body_b.name:
        body_a, body_b = body_b, body_a
    is_body_a_controlled = is_body_controlled(body_a)
    is_body_b_controlled = is_body_controlled(body_b)
    if (not is_body_a_controlled
            and is_body_b_controlled):
        body_a, body_b = body_b, body_a
    return body_a, body_b



@dataclass
class Collision:
    contact_distance_input: float = 0.0
    """
    Distance between the two bodies at which a collision is reported.
    """
    link_a: Body = field(default=None)
    """
    The first body involved in the collision.
    """
    link_b: Body = field(default=None)
    """
    The second body involved in the collision.
    """
    map_P_pa: np.ndarray = field(default=None)
    """
    Pose of the contact point on body A in world coordinates.
    """
    map_P_pb: np.ndarray = field(default=None)
    """
    Pose of the contact point on body B in world coordinates.
    """
    map_V_n_input: np.ndarray = field(default=None)
    a_P_pa: np.ndarray = field(default=None)
    """
    Pose of the contact point on body A in local coordinates.
    """
    b_P_pb: np.ndarray = field(default=None)
    """
    Pose of the contact point on body B in local coordinates.
    """
    data: np.ndarray = field(init=False)
    """
    Data array storing collision information.
    """
    is_external: bool = False
    """
    If the collision was with an external object and not a self-collision.
    """

    _hash_idx: int = 0
    _map_V_n_idx: int = 1
    _map_V_n_slice: slice = slice(1, 4)

    _contact_distance_idx: int = 4
    _new_a_P_pa_idx: int = 5
    _new_a_P_pa_slice: slice = slice(5, 8)

    _new_b_V_n_idx: int = 8
    _new_b_V_n_slice: slice = slice(8, 11)
    _new_b_P_pb_idx: int = 11
    _new_b_P_pb_slice: slice = slice(11, 14)

    _self_data_slice: slice = slice(4, 14)
    _external_data_slice: slice = slice(0, 8)

    def __post_init__(self):
        self.original_link_a = self.link_a
        self.original_link_b = self.link_b

        self.data = np.array([
            self.link_b.__hash__(),  # hash
            0, 0, 1,  # map_V_n

            self.contact_distance_input,
            0, 0, 0,  # new_a_P_pa

            0, 0, 1,  # new_b_V_n
            0, 0, 0,  # new_b_P_pb
        ],
            dtype=float)
        if self.map_V_n_input is not None:
            self.map_V_n = self.map_V_n_input

    @property
    def external_data(self) -> np.ndarray:
        return self.data[:self._new_b_V_n_idx]

    @property
    def self_data(self) -> np.ndarray:
        return self.data[self._self_data_slice]

    @property
    def external_and_self_data(self) -> np.ndarray:
        return self.data[self._external_data_slice]

    @property
    def contact_distance(self) -> float:
        return self.data[self._contact_distance_idx]

    @contact_distance.setter
    def contact_distance(self, value: float):
        self.data[self._contact_distance_idx] = value

    @property
    def link_b_hash(self) -> float:
        return self.data[self._hash_idx]

    @property
    def map_V_n(self) -> np.ndarray:
        """
        Normal vector at the contact point in world coordinates.
        """
        a = self.data[self._map_V_n_slice]
        return np.array([a[0], a[1], a[2], 0])

    @map_V_n.setter
    def map_V_n(self, value: np.ndarray):
        self.data[self._map_V_n_slice] = value[:3]

    @property
    def new_a_P_pa(self):
        a = self.data[self._new_a_P_pa_slice]
        return np.array([a[0], a[1], a[2], 1])

    @new_a_P_pa.setter
    def new_a_P_pa(self, value: np.ndarray):
        self.data[self._new_a_P_pa_slice] = value[:3]

    @property
    def new_b_P_pb(self):
        a = self.data[self._new_b_P_pb_slice]
        return np.array([a[0], a[1], a[2], 1])

    @new_b_P_pb.setter
    def new_b_P_pb(self, value: np.ndarray):
        self.data[self._new_b_P_pb_slice] = value[:3]

    @property
    def new_b_V_n(self):
        a = self.data[self._new_b_V_n_slice]
        return np.array([a[0], a[1], a[2], 0])

    @new_b_V_n.setter
    def new_b_V_n(self, value: np.ndarray):
        self.data[self._new_b_V_n_slice] = value[:3]

    def __str__(self):
        return f'{self.original_link_a}|-|{self.original_link_b}: {self.contact_distance}'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.link_a, self.link_b))

    def __eq__(self, other: Collision):
        return (self.link_a == other.link_a and self.link_b == other.link_b) or (self.link_a == other.link_b and self.link_b == other.link_a)

    def reverse(self):
        return Collision(link_a=self.original_link_b,
                         link_b=self.original_link_a,
                         map_P_pa=self.map_P_pb,
                         map_P_pb=self.map_P_pa,
                         map_V_n_input=-self.map_V_n,
                         a_P_pa=self.b_P_pb,
                         b_P_pb=self.a_P_pa,
                         contact_distance_input=self.contact_distance)

@dataclass
class CollisionCheck:
    """
    A class representing a collision check between two bodies within a specified distance.
    """
    body_a: Body
    """
    The first body involved in the collision check.
    """
    body_b: Body
    """
    The second body involved in the collision check.
    """
    distance: float
    """
    The distance threshold for the collision check.
    """

    def __post_init__(self):
        self.body_a, self.body_b = (self.body_a, self.body_b)

    def __hash__(self):
        return hash((self.body_a, self.body_b))

    def __eq__(self, other: CollisionCheck):
        return self.body_a == other.body_a and self.body_b == other.body_b

    @property
    def bodies(self) -> Tuple[Body, Body]:
        return self.body_a, self.body_b

    def _validate(self, world: World) -> None:
        """Validates the collision check parameters."""
        if self.distance <= 0:
            raise ValueError(f'Distance must be positive, got {self.distance}')

        if self.body_a == self.body_b:
            raise ValueError(f'Cannot create collision check between the same body "{self.body_a.name}"')

        if not self.body_a.has_collision():
            raise ValueError(f'Body {self.body_a.name} has no collision geometry')

        if not self.body_b.has_collision():
            raise ValueError(f'Body {self.body_b.name} has no collision geometry')

        if self.body_a not in world.bodies_with_enabled_collision:
            raise ValueError(f'Body {self.body_a.name} is not in list of bodies with collisions')

        if self.body_b not in world.bodies_with_enabled_collision:
            raise ValueError(f'Body {self.body_b.name} is not in list of bodies with collisions')

        root_chain, tip_chain = world.compute_split_chain_of_connections(self.body_a, self.body_b)
        if all(not isinstance(c, ActiveConnection) for c in chain(root_chain, tip_chain)):
            raise ValueError(f'Relative pose between {self.body_a.name} and {self.body_b.name} is fixed')

    @classmethod
    def create_and_validate(cls, body_a: Body, body_b: Body, distance: float,
                            world: World) -> CollisionCheck:
        """
        Creates a collision check with additional world-context validation.
        Returns None if the check should be skipped (e.g., bodies are linked).
        """
        collision_check = cls(body_a=body_a, body_b=body_b, distance=distance)
        collision_check._validate(world)
        return collision_check



class CollisionDetector(abc.ABC):
    """
    Abstract class for collision detectors.
    """

    @abc.abstractmethod
    def sync_world_model(self) -> None:
        """
        Synchronize the collision checker with the current world model
        """

    @abc.abstractmethod
    def sync_world_state(self) -> None:
        """
        Synchronize the collision checker with the current world state
        """

    @abc.abstractmethod
    def check_collisions(self,
                         collision_matrix: Set[CollisionCheck],
                         buffer: float = 0.05) -> Set[Collision]:
        pass

    @abc.abstractmethod
    def reset_cache(self):
        pass

    def find_colliding_combinations(self, body_combinations: Iterable[Tuple[Body, Body]],
                                    distance: float,
                                    update_query: bool) -> Set[Tuple[Body, Body]]:
        raise NotImplementedError('Collision checking is turned off.')