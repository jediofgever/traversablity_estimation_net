cmake_minimum_required(VERSION 3.15)
project(traversablity_estimation_net)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(include
        ${PCL_INCLUDE_DIRS}
        )

set(dependencies
        rclcpp
        sensor_msgs
        geometry_msgs
        vision_msgs
        rclcpp
        tf2
        tf2_ros
        tf2_geometry_msgs
        tf2_eigen
        pcl_ros
        visualization_msgs
        )

# TOOLS
add_executable(traversablity_smoother_node src/traversablity_smoother_node.cpp)
ament_target_dependencies(traversablity_smoother_node ${dependencies})
target_include_directories(traversablity_smoother_node PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(traversablity_smoother_node ${PCL_LIBRARIES})

install(TARGETS    traversablity_smoother_node
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch 
        DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(${dependencies})

ament_package()
