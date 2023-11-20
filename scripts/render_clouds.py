import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def read_point_cloud(file_path):
    """
    Reads a point cloud file and returns the Open3D point cloud object.
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    return point_cloud

def translate_point_cloud(point_cloud, translation):
    """
    Translates the point cloud by a given vector.
    """
    point_cloud = point_cloud.translate(translation, relative=True)
    return point_cloud

def visualize_side_by_side(pc1, pc2, translation_vector=[1, 0, 0], title1='Original', title2='Prediction'):
    """
    Visualizes two point clouds side by side for comparison, translating one of them.
    """
    # Translate the second point cloud
    pc2_translated = translate_point_cloud(pc2, translation_vector)

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    # Full screen
    vis.create_window(window_name="Point Clouds Comparison", width=1920, height=1080)
    
    #vis.create_window(window_name="Point Clouds Comparison", width=1280, height=720)
    
    # Draw Pink boxes around each point cloud to make them easier to see
    box1 = o3d.geometry.AxisAlignedBoundingBox(min_bound=pc1.get_min_bound(), max_bound=pc1.get_max_bound())
    box2 = o3d.geometry.AxisAlignedBoundingBox(min_bound=pc2_translated.get_min_bound(), max_bound=pc2_translated.get_max_bound())
    box1.color = (1, 0, 1)
    box2.color = (1, 0, 1)
    vis.add_geometry(box1)
    vis.add_geometry(box2)
    
    # Add the first point cloud
    vis.add_geometry(pc1)
    # Add the translated second point cloud
    vis.add_geometry(pc2_translated)
    
    # make the background black
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
 
    # Set the view point
    vis.get_view_control().set_front([0, -0.75, 1])
    vis.get_view_control().set_lookat([11, 0, 0])
    vis.get_view_control().set_up([0, 1, 0])
    vis.get_view_control().set_zoom(0.35)

    
    # Make point size smaller
    vis.get_render_option().point_size = 1.0

    # Render the visualizer
    vis.update_geometry(pc1)
    vis.update_geometry(pc2_translated)
    vis.poll_events()
    vis.update_renderer()
    
    # wait user to close window
    vis.run()

    # Capture image
    #image = vis.capture_screen_float_buffer(do_render=True)
    #plt.imshow(np.asarray(image))
    #plt.title(f"{title1} vs {title2}")
    #plt.axis('off')
    #plt.show()

    # Close the visualizer
    #vis.destroy_window()

# Example usage
# Load your point clouds (replace 'path_to_original.pcd' and 'path_to_prediction.pcd' with your actual file paths)

#iterate through all pcd original and predicted

# get all files under /home/atas/ros2_ws/ starting with original_cloud*
import glob
import os
import copy

# get all files under /home/atas/ros2_ws/ starting with original_cloud*
original_files = glob.glob('/home/atas/ros2_ws/original_cloud*')
predicted_files = glob.glob('/home/atas/ros2_ws/traversable_cloud*')

# sort the files
original_files.sort()
predicted_files.sort()

# iterate through all files
for i in range(len(original_files)):
    # load the original and predicted point clouds
    original_pc = read_point_cloud(original_files[i])
    predicted_pc = read_point_cloud(predicted_files[i])
    
    # Translate the second point cloud to the right for side-by-side comparison
    translate_vector = [22, 0, 0] # Adjust this vector as needed

    # Visualize the point clouds side by side
    visualize_side_by_side(original_pc, predicted_pc, translate_vector)


 

