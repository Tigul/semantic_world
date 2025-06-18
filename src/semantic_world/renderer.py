from typing import Dict, List

import numpy as np

from .world import World
from .geometry import Mesh


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
                if isinstance(collision_shape, Mesh):
                    o3d_mesh = open3d.io.read_triangle_mesh(collision_shape.filename)
                    triangles = open3d.t.geometry.TriangleMesh().from_legacy(mesh_legacy=o3d_mesh) # TODO Add transformation
                    self.bodies_to_scene[body] = scene.add_triangles(triangles)
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
        mask =  self.cast_rays_in_scene(camera_pose, target_pose)["geometry_ids"].numpy()
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
        import open3d
        rays = self.scene.create_rays_pinhole(
            fov_deg=90,
            center=camera_pose[:3],
            eye=target_pose[:3],
            up=[0, 0, -1],
            width_px=640,
            height_px=480
        )
        ans = self.scene.cast_rays(rays)
        return ans
