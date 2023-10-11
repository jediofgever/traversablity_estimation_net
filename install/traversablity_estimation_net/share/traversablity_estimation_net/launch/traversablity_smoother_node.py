from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    traversablity_smoother_node = Node(
        package="vox_nav_misc",
        executable="traversablity_smoother_node",
        name="traversablity_smoother_node",
        output="screen",
        remappings=[
            ("image", "/pointnet/traversability/crop_boxes"),
            ("points", "/pointnet/traversability/map_local"),
        ],
    )

    ld = LaunchDescription()

    ld.add_action(traversablity_smoother_node)

    return ld
