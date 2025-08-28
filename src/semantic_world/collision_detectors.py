from dataclasses import dataclass, field
from typing import Set, List

from trimesh.collision import CollisionManager

from .collisions import CollisionDetector, CollisionCheck, Collision
from .world import World


@dataclass
class TrimeshCollisionChecker(CollisionDetector):
    world: World
    """
    World model to perform collision checking on
    """

    collision_manager: CollisionManager = field(default_factory=CollisionManager, init=False)
    """
    The collision manager from trimesh to handle collision detection
    """

    def sync_world_model(self) -> None:
        """
        Synchronize the collision checker with the current world model
        """
        for body in self.world.bodies_with_collisions:
            self.collision_manager.add_object(body.name.name, body.combined_collision_mesh, body.global_pose.to_np())

    def sync_world_state(self) -> None:
        """
        Synchronize the collision checker with the current world state
        """
        for body in self.world.bodies_with_collisions:
            self.collision_manager.set_transform(body.name.name, body.global_pose.to_np())

    def check_collisions(self,
                         collision_matrix: Set[CollisionCheck] = None,
                         buffer: float = 0.05) -> Set[Collision]:
        collisions = self.collision_manager.in_collision_internal(return_names=True, return_data=True)
        collision_pairs = [(self.world.get_kinematic_structure_entity_by_name(pair[0]), self.world.get_kinematic_structure_entity_by_name(pair[1])) for pair in collisions[1]]
        result_set = set()
        for collision_check in collision_matrix or []:
            if (collision_check.body_a, collision_check.body_b) in collision_pairs or (collision_check.body_b, collision_check.body_a) in collision_pairs:

                for data in collisions[2]:
                    if data.names == {collision_check.body_a.name.name, collision_check.body_b.name.name} or data.names == {collision_check.body_b.name.name, collision_check.body_a.name.name}:
                        result_set.add(Collision(0.0, collision_check.body_a, collision_check.body_b, map_P_pa=data.point))
        return result_set
    def reset_cache(self):
        pass
