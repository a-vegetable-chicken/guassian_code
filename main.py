import blenderproc as bproc
import numpy as np
import os

data_path = "/scratch_net/biwidl311/wty/HM3D/"
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
            elif key == 'm_colorWidth':
                intrinsics["color_width"] = value
            elif key == 'm_colorHeight':
                intrinsics["color_height"] = value


# Load a random Matterport3D room
[objects, floor]= bproc.loader.load_matterport3d(data_path)

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

color_k = intrinsics["color"][:3, :3] 
depth_k = intrinsics["depth"][:3, :3]

width = int(intrinsics["color_width"])
height = int(intrinsics["color_height"])
bproc.camera.set_intrinsics_from_K_matrix(
    K_matrix=color_k,
    image_width=width,
    image_height=height
)

bproc.renderer.set_light_bounces(
    diffuse_bounces=200,
    glossy_bounces=200,
    max_bounces=200,
    transmission_bounces=200
)
bproc.renderer.set_output_format(enable_transparency=True)


bproc.renderer.enable_depth_output(
    activate_antialiasing=False,
    depth_layers=np.inf
)

render_data = bproc.renderer.render()

depth_shift = 1000  
render_data["depth"] = [d * depth_shift for d in render_data["depth"]]


output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

bproc.writer.write_hdf5(
    output_dir,
    render_data,
    colors=render_data["colors"],
    depths=render_data["depth"],
    append_to_existing_output=False,
    custom_attributes={
        "intrinsics": {
            "color": color_k.tolist(),
            "depth": depth_k.tolist()
        },
        "resolution": {
            "color": (width, height),
            "depth": (int(intrinsics.get("m_depthWidth", 224)), 
            int(intrinsics.get("m_depthHeight", 172))
        }
    }
)