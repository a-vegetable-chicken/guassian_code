import blenderproc as bproc
import numpy as np
import os

data_path = "/scratch_net/biwidl311/wty/HM3D/"
info_txt_path = "./_info.txt"
'''
intrinsics = {}
with open(info_txt_path, "r") as file:
    for line in file:
        parts = line.strip().split(" = ")
        if len(parts) == 2:

            key, value = parts

            if key == "m_calibrationColorIntrinsic":
                intrinsics["color"] = np.array(value.split(), dtype=float).reshape(4, 4)
            elif key == "m_calibrationDepthIntrinsic":
                intrinsics["depth"] = np.array(value.split(), dtype=float).reshape(4, 4)
            elif key == 'm_colorWidth':
                intrinsics["color_width"] = value
            elif key == 'm_colorHeight':
                intrinsics["color_height"] = value


bproc.camera.set_intrinsics_from_K_matrix(
    intrinsics["color"][:3, :3]
    int(intrinsics["color_width"])
    int(intrinsics["color_height"])
)
'''
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
    #print(bproc.camera.get_camera_pose())
    poses += 1
    if poses == 5:
        break

p = bproc.camera.get_camera_poses()
print(p)
    

'''
bproc.renderer.set_max_amount_of_samples(50)
data = bproc.renderer.render()

output_dir = '../output'
bproc.writer.write_png(
    output_dir, 
    data, 
    color_depth="8"  
)
'''