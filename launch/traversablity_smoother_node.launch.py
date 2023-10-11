from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    traversablity_smoother_node = Node(
        package="traversablity_estimation_net",
        executable="traversablity_smoother_node",
        name="traversablity_smoother_node",
        output="screen",
        remappings=[
            ("boxes", "/pointnet/traversability/crop_boxes"),
            ("points", "/pointnet/traversability/map_local"),
        ],
        parameters=[
            {"use_sim_time": True,
             "kdtree_search_radius": 0.5,
             "cropped_cloud_voxel_size": 0.2,
             },
            
 
        ],
    )

    ld = LaunchDescription()

    ld.add_action(traversablity_smoother_node)

    return ld