from copy import copy
from typing import Dict, List

import numpy as np

from .world import World
from .geometry import Mesh, Box, Cylinder, Sphere


class Renderer:
    def __init__(self, world: World):
        """
        Renderer for the world using Open3D's raycasting capabilities. Is able to create segmentation masks and depth maps
        from the world geometry.
        :param world: The world to render.
        """
        self.world = world
        self.scene = None
        self.bodies_to_scene = {}
        self.world_model_version = -1
        self.world_state_version = -1

    def create_raycast_scene(self):
        """
        Create a raycast scene from the world.
        """
        if self.world._model_version == self.world_model_version and self.world._state_version == self.world_state_version:
            return
        import open3d
        scene = open3d.t.geometry.RaycastingScene()
        for body in self.world.bodies:
            for collision_shape in body.collision:
                pose_transform = self.world.compute_forward_kinematics_np(self.world.root, body)
                if isinstance(collision_shape, Mesh):
                    self.bodies_to_scene[body] = scene.add_triangles(
                        copy(collision_shape.mesh).transform(pose_transform))
                elif isinstance(collision_shape, Box):
                    box = open3d.geometry.TriangleMesh().create_box(width=collision_shape.scale.x,
                                                                    height=collision_shape.scale.y,
                                                                    depth=collision_shape.scale.z).transform(
                        pose_transform)
                    tensor_box = open3d.t.geometry.TriangleMesh.from_legacy(box)
                    self.bodies_to_scene[collision_shape] = scene.add_triangles(tensor_box)
                elif isinstance(collision_shape, Cylinder):
                    self.bodies_to_scene[body] = scene.add_triangles(
                        open3d.t.geometry.TriangleMesh().create_cylinder(radius=collision_shape.width / 2,
                                                                         height=collision_shape.height).transform(
                            pose_transform))
                elif isinstance(collision_shape, Sphere):
                    self.bodies_to_scene[body] = scene.add_triangles(
                        open3d.t.geometry.TriangleMesh().create_sphere(radius=collision_shape.radius).transform(
                            pose_transform))
        self.scene = scene
        self.world_model_version = self.world._model_version
        self.world_state_version = self.world._state_version

    def create_segmentation_mask(self, camera_pose: List[float], target_pose: List[float]):
        """
        Create a segmentation mask from the camera pose to the target pose. Returns an array where each pixel
        corresponds to a body in the world if it is visible from the camera pose.
        :param camera_pose: The pose of the camera as a list of floats [x, y, z], in world coordinates.
        :param target_pose: The pose of the target as a list of floats [x, y, z], in world coordinates.
        :return: An array of the visible bodies in the world
        """
        mask = self.cast_rays_in_scene(camera_pose, target_pose)["geometry_ids"].numpy()
        vectorized_map = np.vectorize(lambda x: self.bodies_to_scene[x])
        return vectorized_map(mask)

    def create_depth_map(self, camera_pose: List[float], target_pose: List[float]) -> np.ndarray:
        """
        Create a depth map from the camera pose to the target pose.
        :param camera_pose: The pose of the camera as a list of floats [x, y, z], in world coordinates.
        :param target_pose: The pose of the target as a list of floats [x, y, z], in world coordinates.
        :return: A numpy array of shape (height, width) containing the depth values.
        """
        return self.cast_rays_in_scene(camera_pose, target_pose)["t_hit"].numpy()

    def cast_rays_in_scene(self, camera_pose: List[float], target_pose: List[float]) -> Dict[str, np.ndarray]:
        """
        Cast rays in the scene.
        :param camera_pose: The pose of the camera as a list of floats [x, y, z], in world coordinates.
        :param target_pose: The pose of the target as a list of floats [x, y, z], in world coordinates.
        :return: A dictionary containing the results of the raycasting, including 'geometry_ids' and 't_hit'.
        """
        self.create_raycast_scene()
        rays = self.scene.create_rays_pinhole(
            fov_deg=90,
            center=target_pose,
            eye=camera_pose,
            up=[0, 0, -1],
            width_px=640,
            height_px=480
        )
        return self.scene.cast_rays(rays)

    def calculate_signed_distance(self, min_pose: np.ndarray, max_pose: np.ndarray, steps: int = 32) -> np.ndarray:
        """
        Calculate the signed distance field for a given bounding box defined by min and max poses. The bounding box is
        discretized into a grid of points by the steps parameter, and the signed distance is computed for each point.
        :param min_pose: The minimum pose of the bounding box as a numpy array of shape (3,).
        :param max_pose: The maximum pose of the bounding box as a numpy array of shape (3,).
        :param steps: The number of steps to divide the bounding box into, default is 32.
        :return: A numpy array containing the signed distance values.
        """
        self.create_raycast_scene()
        xyz_range = np.linspace(min_pose, max_pose, num=steps)

        # query_points is a [steps,steps,steps,3] array ..
        query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

        # signed distance is a [32,32,32] array
        signed_distance = self.scene.compute_signed_distance(query_points).numpy()
        return signed_distance
