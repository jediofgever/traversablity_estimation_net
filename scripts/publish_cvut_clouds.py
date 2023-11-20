
import os
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import copy 
from sensor_msgs_py.point_cloud2 import read_points_numpy, create_cloud

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('point_cloud_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'cvut_point_cloud', 10)
        self.timer = self.create_timer(3.0, self.timer_callback)
        self.pcd_files = self.get_pcd_files('/home/atas/subtdata.felk.cvut.cz/robingas/data/traversability_estimation/TraversabilityDataset/supervised/clouds/destaggered_points_colored/')  # Replace with your directory path
        self.pcd_files_iter = iter(self.pcd_files)

    def get_pcd_files(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pcd')]

    def timer_callback(self):
        try:
            pcd_file = next(self.pcd_files_iter)
            pcd = o3d.io.read_point_cloud(pcd_file)

            # Convert Open3D point cloud to ROS2 PointCloud2 message
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                
                colors = np.asarray(pcd.colors)
                
                cloud_npy = np.asarray(copy.deepcopy(pcd.points))

                n_points = len(cloud_npy[:, 0])
                data = np.zeros(n_points, dtype=[
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('rgb', np.uint32)
                ])

                BIT_MOVE_16 = 2**16
                BIT_MOVE_8 = 2**8

                rgb_npy = np.asarray(copy.deepcopy(pcd.colors))
                rgb_npy = np.floor(rgb_npy*255)  # nx3 matrix
                rgb_npy = rgb_npy[:, 0] * BIT_MOVE_16 + \
                    rgb_npy[:, 1] * BIT_MOVE_8 + rgb_npy[:, 2]
                rgb_npy = rgb_npy.astype(np.uint32)

                data['x'] = cloud_npy[:, 0]
                data['y'] = cloud_npy[:, 1]
                data['z'] = cloud_npy[:, 2]
                data['rgb'] = rgb_npy

                fields = []
                fields.append(PointField(
                    name="x",
                    offset=0,
                    datatype=PointField.FLOAT32, count=1))
                fields.append(PointField(
                    name="y",
                    offset=4,
                    datatype=PointField.FLOAT32, count=1))
                fields.append(PointField(
                    name="z",
                    offset=8,
                    datatype=PointField.FLOAT32, count=1))
                fields.append(PointField(
                    name="rgb",
                    offset=12,
                    datatype=PointField.UINT32, count=1))
                point_step = 16
                
                header = Header()
                header.frame_id = "base_link"

                final_ros_cloud = create_cloud(header=header,
                                            fields=fields,
                                            points=data)
                
                self.publisher_.publish(final_ros_cloud)
                self.get_logger().info(f'Publishing {pcd_file}')
        
            else:
                data = points
                fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(['x', 'y', 'z'])]

                header = Header()
                header.frame_id = "base_link"
                cloud_msg = point_cloud2.create_cloud(header, fields, data)

                self.publisher_.publish(cloud_msg)
                self.get_logger().info(f'Publishing {pcd_file}')

        except StopIteration:
            self.get_logger().info('All point clouds have been published')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    point_cloud_publisher = PointCloudPublisher()
    rclpy.spin(point_cloud_publisher)
    point_cloud_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    