import open3d as o3d
from natsort import natsorted
import os
import time

input_dir = "/home/atas/unity_data/data/raw/test"
input_files = natsorted(os.listdir(input_dir))
    
# read first ply file
pcd = o3d.io.read_point_cloud(os.path.join(input_dir, input_files[0]))
vis = o3d.visualization.Visualizer()

# iterate through remaining files     
for input_file in input_files[1:]:
    input_file = os.path.join(input_dir, input_file)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960) 
    vis.clear_geometries() 
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
    vis.add_geometry(pcd)     

    vis.get_render_option().show_coordinate_frame = True
    
    print("Printing " + input_file)
    vis.run()
    
    # ask user whether this file should be deleted
    delete = input("Delete file? (y/N): ")
    if delete == "y":
        os.remove(input_file)
        print("Deleted " + input_file)

