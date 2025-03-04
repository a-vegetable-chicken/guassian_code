import blenderproc as bproc
import numpy as np
import os

data_path = "/scratch_net/biwidl311/wty/HM3D/v1/scans/8194nk5LbLH/8194nk5LbLH/matterport_mesh/9266ab00ab6744348efa7afe13b3db9f/9266ab00ab6744348efa7afe13b3db9f.obj"
info_txt_path = "./_info.txt"

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

# Load a random Matterport3D room
objects, floor= bproc.loader.load_matterport3d(data_path)

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

    
color_intr_3x3 = intrinsics["color"][:3, :3]
bproc.camera.set_intrinsics_from_K_matrix(color_intr_3x3, image_width=960, image_height=540)

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

data = bproc.renderer.render()


output_dir = "../output"

os.makedirs(output_dir, exist_ok=True)

bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=False)

bproc.writer.write_png(output_dir, data, color_depth="8")

print(f"render {poses} and output to: {output_dir}")
