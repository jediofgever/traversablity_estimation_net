import open3d as o3d
from natsort import natsorted
import os
import time
import numpy as np
input_dir = "/home/atas/unity_data/data/raw/test"
input_files = natsorted(os.listdir(input_dir))
    
# read first ply file
pcd = o3d.io.read_point_cloud(os.path.join(input_dir, input_files[0]))
vis = o3d.visualization.Visualizer()

# iterate through remaining files     
for input_file in input_files[1:]:
    
    input_file = os.path.join(input_dir, input_file)
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
    
    voxel_size = np.random.uniform(0.05, 0.25)
    down_cloud = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd.points).astype(np.float32)

    noise = np.random.normal(0.01, 0.04, points.shape)
    points = points + noise
    pcd.points = o3d.utility.Vector3dVector(points)
            
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960) 
    vis.clear_geometries()     
    vis.add_geometry(pcd)     
    vis.get_render_option().show_coordinate_frame = True
    vis.run()
    
    print("Printing " + input_file)


