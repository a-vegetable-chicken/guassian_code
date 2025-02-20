import blenderproc as bproc
import numpy as np

data_path = "/scratch/wty/hm3d-minival-glb-v0.2"

# Load a random Matterport3D room
objects, floor = bproc.loader.load_matterport3d(data_path)

print(f"Loaded {len(objects)} objects.")
print(f"Floor object: {floor}")
