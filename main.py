import blenderproc as bproc
import numpy as np

data_path = "/scratch/wty/hm3d-minival-glb-v0.2/00801-HaxA7YrQdEC/HaxA7YrQdEC.glb"

# Load a random Matterport3D room
objects, floor = bproc.loader.load_matterport3d(data_path)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([objects, floor])

poses = 0
for try_counter in range(10000):
    location = bproc.sampler.upper_region([floor], min_height=1.5, max_height=1.8)
    # Check that there is no object between the sampled point and the floor
    _, _, _, _, hit_object, _ = bproc.object.scene_ray_cast(location, [0, 0, -1])
    if hit_object != floor:
        continue

    # Sample rotation (fix around X and Y axis)
    rotation = np.random.uniform([1.2217, 0, 0], [1.2217, 0, 2 * np.pi])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    # Check that there is no obstacle in front of the camera closer than 1m
    if not bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 1.0, "no_background": True},
                                                       bvh_tree, sqrt_number_of_rays=20):
        continue

    # If all checks were passed, add the camera pose
    bproc.camera.add_camera_pose(cam2world_matrix)
    poses += 1
    if poses == 5:
        break