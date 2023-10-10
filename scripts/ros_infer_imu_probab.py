
import time
import copy

import open3d as o3d
import torch
import numpy as np

import rclpy
import rclpy.qos
from rclpy.node import Node
from geometry_msgs.msg import Transform, TransformStamped, Pose, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import PointField, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy, create_cloud
from std_msgs.msg import Header
import sensor_msgs.msg as sensor_msgs
from vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose
from std_msgs.msg import Float32MultiArray


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from dataset import pc_normalize
from pointnet_curv import PointnetCurv
from datasetIMU import compute_curvature_static

import matplotlib.pyplot as plt

def scalar_to_rgb(scalar):
    # Invert the scalar value (1 - scalar) to map high values to low hue (green)
    inverted_scalar = 1 - scalar

    # Map the inverted scalar value to the HSV color space
    hue = inverted_scalar * 240  # 0 to 1 maps to 0° to 240° (green to red)
    saturation = 1.0   # Full saturation
    value = 1.0        # Full brightness

    # Convert HSV to RGB
    hue /= 60.0
    chroma = value * saturation
    x = chroma * (1 - abs(hue % 2 - 1))
    
    if 0 <= hue < 1:
        r, g, b = chroma, x, 0
    elif 1 <= hue < 2:
        r, g, b = x, chroma, 0
    elif 2 <= hue < 3:
        r, g, b = 0, chroma, x
    elif 3 <= hue < 4:
        r, g, b = 0, x, chroma
    elif 4 <= hue < 5:
        r, g, b = x, 0, chroma
    else:
        r, g, b = chroma, 0, x

    m = value - chroma
    r, g, b = r + m, g + m, b + m

    # Scale RGB values to the range [0, 1]
    r /= 1.0
    g /= 1.0
    b /= 1.0

    return r, g, b


class PCDSubPubNode(Node):
    """ Node for subscribing to a point cloud and publishing the traversability map
        Uses a pretrained PointNet model to infer the traversability of the local map
        Publishes the traversability map as a point cloud and crop boxes as a Detection3DArray 

    Args:
        Node (rclpy): This is a ROS 2 node
    """

    def __init__(self):
        super().__init__('pcd_subsriber_node')
        
        self.local_map_topic_name = '/modified_map'
        self.traversablity_detection_topic_name = '/pointnet/traversability/map_local'
        self.traversablity_crop_boxes_topic_name = '/pointnet/traversability/crop_boxes'
        self.get_logger().info('Subscribing to ' + self.local_map_topic_name)
        self.get_logger().info('Publishing to ' +
                               self.traversablity_detection_topic_name)
        self.get_logger().info('Publishing to ' +
                               self.traversablity_crop_boxes_topic_name)

        self.model_path = 'weights/epoch_550.pt'
        self.batch_size = 256
        self.use_sim_time = True
        self.use_two_directions = False

        # config for the crop boxes and traversability map
        # crop the cloud to the region of interest
        self.min_corner = [-4, -4, -4.5]
        self.max_corner = [4, 4, 1.5]
        self.x_box_size = 2.0
        self.y_box_size = self.x_box_size / 2.0
        self.min_points = 3
        self.x_step_size = self.x_box_size / 2.0
        self.y_step_size = self.y_box_size / 2.0
        self.kdtree_radius = 0.3
        self.cropped_cloud_downsample_size = 0.2
        
        # size of imu data array (1, 13)
        self.latest_imu_data = np.zeros((13, 1))

        # Print config parameters
        self.get_logger().info('Model path: ' + self.model_path)
        self.get_logger().info('Batch size: ' + str(self.batch_size))
        self.get_logger().info('Use sim time: ' + str(self.use_sim_time))
        self.get_logger().info('Min corner: ' + str(self.min_corner))
        self.get_logger().info('Max corner: ' + str(self.max_corner))
        self.get_logger().info('X box size: ' + str(self.x_box_size))
        self.get_logger().info('Y box size: ' + str(self.y_box_size))
        self.get_logger().info('X step size: ' + str(self.x_step_size))
        self.get_logger().info('Y step size: ' + str(self.y_step_size))
        self.get_logger().info('Min points: ' + str(self.min_points))
        self.get_logger().info('kd tree radius: ' + str(self.kdtree_radius))

        self.from_frame_rel = 'odom'
        self.to_frame_rel = 'base_link'

        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,                            # Msg type
            self.local_map_topic_name,                          # topic
            self.listener_callback,                             # Function to call
            1       # QoS
        )
        
        self.imu_subscriber = self.create_subscription(Float32MultiArray, 'imu_info', self.imu_callback, 1)

        self.obstacle_pcd_publisher = self.create_publisher(
            sensor_msgs.PointCloud2,
            self.traversablity_detection_topic_name,
            # 1)
            rclpy.qos.qos_profile_sensor_data)

        self.box_publisher = self.create_publisher(
            Detection3DArray,
            self.traversablity_crop_boxes_topic_name,
            1)

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # set use_sim_time to true to use the simulation clock
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, self.use_sim_time)])
        self.net = PointnetCurv()

        # load the model
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()
        self.net.cuda()
        self.net.zero_grad()
        self.get_logger().info('Model loaded')

    def imu_callback(self, imu):
        # loop through the imu data and populate the latest_imu_data array
        for i in range(0, len(imu.data)):
            self.latest_imu_data[i, 0] = imu.data[i]
        
            
    def listener_callback(self, msg: sensor_msgs.PointCloud2):
        """ Callback function for the subscriber
            Transforms the point cloud to the base_link frame and then calls the infer function
            publishes the traversability map and crop boxes
        """
        self.get_logger().info('Received point cloud, Infering...')

        try:
            # The local cloud is in "map" frame
            # We need to transform it to "base_link" frame
            trans = self.buffer.lookup_transform(
                self.to_frame_rel,
                self.from_frame_rel,
                rclpy.time.Time())

            # Now create a open3d point cloud
            numpy_array_points = read_points_numpy(
                msg, field_names=('x', 'y', 'z'))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(numpy_array_points)

            self.infer(pcd, trans, msg.header)

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.to_frame_rel} to {self.from_frame_rel}: {ex}')
            return

    def batch(self, iterable, n=1):
        """creates batches of size n from an iterable, typically a list
        e.g. if input is [1,2,3,4,5,6,7,8,9] and n=3, output is [[1,2,3],[4,5,6],[7,8,9]]
        e.g. if input is [1,2,3,4,5,6,7,8,9] and n=4, output is [[1,2,3,4],[5,6,7,8],[9]]

        Args:
            iterable (list): a list 
            n (int, optional): the length batch. Defaults to 1.

        Yields:
            batched list: with each elemnts havce size n or less
        """
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def populate_and_publish_boxes(self,  boxes, header):
        """Populates the Detection3DArray message and publishes it
           The point cloud is cropped to an inflated robot footprint and fed 
           to the PointNet model. The output of the model is a regressed values between 0 and 1
           For debugging purposes, the crop boxes publish as a Detection3DArray message
        """
        detection_array = Detection3DArray()
        hyp = ObjectHypothesisWithPose()
        detection_array.header = header
        detection_array.detections = []

        for box in boxes:

            min_bound = box.get_min_bound()
            max_bound = box.get_max_bound()

            center = Pose()
            center.position.x = box.get_center()[0]
            center.position.y = box.get_center()[1]
            center.position.z = box.get_center()[2]
            center.orientation.w = 1.0

            size = Vector3()
            size.x = max_bound[0] - min_bound[0]
            size.y = max_bound[1] - min_bound[1]
            size.z = max_bound[2] - min_bound[2]

            detection = Detection3D()
            detection.header = header
            detection.bbox.size = size
            detection.bbox.center = center
            detection.results = [hyp]
            detection_array.detections.append(detection)

        self.box_publisher.publish(detection_array)

    def infer(self,
              pcd: o3d.geometry.PointCloud,
              trans: TransformStamped,
              header: Header):

        # calculate the time taken to transform the cloud
        start = time.time()
        # Transfrom cloud to base_link frame
        T = np.eye(4)
        T[:3, :3] = pcd.get_rotation_matrix_from_quaternion((trans.transform.rotation.w, trans.transform.rotation.x,
                                                             trans.transform.rotation.y, trans.transform.rotation.z))
        T[0, 3] = trans.transform.translation.x
        T[1, 3] = trans.transform.translation.y
        T[2, 3] = trans.transform.translation.z
        pcd_transformed = pcd.transform(T)

        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            self.min_corner, self.max_corner))
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=15))
                
        # downsample the cloud
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        
        # paint the cloud with zeros 
        pcd.paint_uniform_color([0, 0, 0])
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)    
        
        x_dist = abs(self.max_corner[0] - self.min_corner[0])
        y_dist = abs(self.max_corner[1] - self.min_corner[1])

        geometries = []
        boxes = []
        data_list = []
        non_normalized_points = []

        for x in range(0, int(x_dist / self.x_step_size)):

            for y in range(0, int(y_dist / self.y_step_size)):

                current_min_corner = [
                    self.min_corner[0] + self.x_step_size * x,
                    self.min_corner[1] + self.y_step_size * y, -4.5,
                ]

                current_max_corner = [
                    current_min_corner[0] + self.x_box_size,
                    current_min_corner[1] + self.y_box_size,  1.5,
                ]

                this_box = o3d.geometry.AxisAlignedBoundingBox(
                    current_min_corner, current_max_corner
                )

                boxes.append(this_box)

                cropped_pcd = pcd.crop(this_box)

                if len(cropped_pcd.points) < self.min_points:
                    continue
                else:
                    normals = np.asarray(cropped_pcd.normals).astype(np.float32)
                    points = np.asarray(cropped_pcd.points).astype(np.float32)
                    non_normalized_crop = copy.deepcopy(points)
                    non_normalized_points.append(torch.tensor(non_normalized_crop))

                    points = pc_normalize(points)
                    points = np.hstack((points, normals))

                    # if this box is centered at the back of the car, we rotate cloud by 180 degrees
                    if (current_max_corner[0] + current_min_corner[0]) / 2 < 0:
                        # rotate by 180 degrees
                        if self.use_two_directions:
                            deep_copy = copy.deepcopy(cropped_pcd)
                            points = np.asarray(
                                deep_copy.points).astype(np.float32)
                            points = pc_normalize(points)
                            deep_copy.points = o3d.utility.Vector3dVector(
                                points)
                            R = deep_copy.get_rotation_matrix_from_xyz(
                                (0, 0, np.pi))
                            deep_copy.rotate(R, center=(0, 0, 0))
                            points = np.asarray(
                                deep_copy.points).astype(np.float32)

                    points = torch.tensor(points)
                    points = points.reshape(
                        (points.shape[0], points.shape[1], 1))
                    data_list.append(points)
                    geometries.append(cropped_pcd)

        header.frame_id = self.to_frame_rel

        self.populate_and_publish_boxes(boxes, header)

        crop_index = 0
        final_cloud = o3d.geometry.PointCloud()
        final_cloud_points = []
        final_cloud_colors = []
        
        # reset all the normals to zero in pcd 
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
        colormap = plt.get_cmap('cividis')
        norm = plt.Normalize(vmin=0, vmax=1)
        # use x component of normals to store the traversability values

        # measure net inference time
        for x in self.batch(data_list, self.batch_size):
            batches = []
            batch_index = 0
            imu_data = []
            for sample in x:
                batches.append(torch.ones((sample.size(dim=0)),
                               dtype=torch.long) * batch_index)
                
                imu_data.append(torch.tensor(self.latest_imu_data, dtype=torch.float32))
                
                batch_index = batch_index + 1

            batches = torch.cat(batches, dim=0).cuda()
            points = torch.cat(x, dim=0).cuda()
            
            # add imu data as torch tensor float32
            imu_data = torch.cat(imu_data, dim=1).cuda()
            
            print("imu data shape: ", imu_data.shape)

            predictions = self.net(points, imu_data, batches).cpu().detach().numpy()
            
            

            for crop_index, p in enumerate(predictions):
                p = np.clip(p, 0, 1)
                downsampled_crop = geometries[crop_index].voxel_down_sample(voxel_size=self.cropped_cloud_downsample_size)
                non_normalized_crop = np.asarray(downsampled_crop.points)
                pcd_normals = np.asarray(pcd.normals)
                pcd_colors = np.asarray(pcd.colors)
            

                for point in non_normalized_crop:
                    # Find neighbors in the original cloud kdtree
                    [k, idx, _] = pcd_tree.search_radius_vector_3d(point, self.kdtree_radius)

                    if len(idx) > 0:
                        mean_x_normal = np.mean(pcd_normals[idx], axis=0)[0]
                        
                        #continue
                        updated_x_normal = (len(idx) * mean_x_normal + p[0]) / (len(idx) + 1)

                        # Update normals and colors
                        pcd_normals[idx, 0] = updated_x_normal
                        
                        r, g, b = scalar_to_rgb(updated_x_normal)

                        
                        pcd_colors[idx[1:], :] = [r,g,b]
        
        if len(non_normalized_points) == 0:
            self.get_logger().info("No points in the cloud")
            return

        final_cloud.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points, dtype=np.float32))
        final_cloud.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors, dtype=np.float32))
        
        cloud_npy = np.asarray(copy.deepcopy(final_cloud.points))

        n_points = len(cloud_npy[:, 0])
        data = np.zeros(n_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])

        BIT_MOVE_16 = 2**16
        BIT_MOVE_8 = 2**8

        rgb_npy = np.asarray(copy.deepcopy(final_cloud.colors))
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

        final_ros_cloud = create_cloud(header=header,
                                       fields=fields,
                                       points=data)

        self.get_logger().info("Inference time: " + str(time.time() - start))
        self.obstacle_pcd_publisher.publish(final_ros_cloud)


def main():
    rclpy.init()
    node = PCDSubPubNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
