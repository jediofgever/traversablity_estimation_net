
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

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from dataset import pc_normalize
from pointnet import PointNet


class PCDSubPubNode(Node):
    """ Node for subscribing to a point cloud and publishing the traversability map
        Uses a pretrained PointNet model to infer the traversability of the local map
        Publishes the traversability map as a point cloud and crop boxes as a Detection3DArray 

    Args:
        Node (rclpy): This is a ROS 2 node
    """

    def __init__(self):
        super().__init__('pcd_subsriber_node')

        self.local_map_topic_name = '/lio_sam/mapping/map_local'
        self.traversablity_detection_topic_name = '/pointnet/traversability/map_local'
        self.traversablity_crop_boxes_topic_name = '/pointnet/traversability/crop_boxes'
        self.get_logger().info('Subscribing to ' + self.local_map_topic_name)
        self.get_logger().info('Publishing to ' +
                               self.traversablity_detection_topic_name)
        self.get_logger().info('Publishing to ' +
                               self.traversablity_crop_boxes_topic_name)

        self.model_path = 'weights/epoch_200.pt'
        self.batch_size = 128
        self.use_sim_time = True
        self.use_two_directions = False

        # config for the crop boxes and traversability map
        # crop the cloud to the region of interest
        self.min_corner = [-10, -10, -4.5]
        self.max_corner = [10, 10, 1.5]
        self.x_step_size = 1.5
        self.y_step_size = 2.0*self.x_step_size / 3.0
        self.min_points = 3

        # Print config parameters
        self.get_logger().info('Model path: ' + self.model_path)
        self.get_logger().info('Batch size: ' + str(self.batch_size))
        self.get_logger().info('Use sim time: ' + str(self.use_sim_time))
        self.get_logger().info('Min corner: ' + str(self.min_corner))
        self.get_logger().info('Max corner: ' + str(self.max_corner))
        self.get_logger().info('X step size: ' + str(self.x_step_size))
        self.get_logger().info('Y step size: ' + str(self.y_step_size))
        self.get_logger().info('Min points: ' + str(self.min_points))

        self.from_frame_rel = 'map'
        self.to_frame_rel = 'base_link'

        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,                            # Msg type
            self.local_map_topic_name,                          # topic
            self.listener_callback,                             # Function to call
            1       # QoS
        )

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
        self.net = PointNet()

        # load the model
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()
        self.net.cuda()
        self.net.zero_grad()
        self.get_logger().info('Model loaded')

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

        x_dist = abs(self.max_corner[0] - self.min_corner[0])
        y_dist = abs(self.max_corner[1] - self.min_corner[1])

        geometries = []
        boxes = []
        data_list = []
        non_normalized_points = []

        for x in range(0, int(x_dist / self.x_step_size+1)):

            for y in range(0, int(y_dist / self.y_step_size+1)):

                current_min_corner = [
                    self.min_corner[0] + self.x_step_size * x,
                    self.min_corner[1] + self.y_step_size * y, -4.5,
                ]

                current_max_corner = [
                    current_min_corner[0] + self.x_step_size,
                    current_min_corner[1] + self.y_step_size,  1.5,
                ]

                this_box = o3d.geometry.AxisAlignedBoundingBox(
                    current_min_corner, current_max_corner
                )

                boxes.append(this_box)

                cropped_pcd = pcd.crop(this_box)

                if len(cropped_pcd.points) < self.min_points:
                    continue
                else:

                    points = np.asarray(cropped_pcd.points).astype(np.float32)
                    non_normalized_crop = copy.deepcopy(points)
                    non_normalized_points.append(
                        torch.tensor(non_normalized_crop))

                    points = pc_normalize(points)

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

        for x in self.batch(data_list, self.batch_size):
            batches = []
            batch_index = 0
            for sample in x:
                batches.append(torch.ones((sample.size(dim=0)),
                               dtype=torch.long) * batch_index)
                batch_index = batch_index + 1

            batches = torch.cat(batches, dim=0).cuda()
            points = torch.cat(x, dim=0).cuda()

            predictions = self.net(points, batches).cpu().detach().numpy()

            for p in predictions:
                p = np.clip(p, 0, 1)
                if p[0] < 0.2:
                    geometries[crop_index].paint_uniform_color(
                        [0, 1, 0])
                else:
                    geometries[crop_index].paint_uniform_color([1, 0, 0])

                curr_points = np.asarray(
                    geometries[crop_index].points, dtype=np.float32)
                curr_colors = np.asarray(
                    geometries[crop_index].colors, dtype=np.float32)

                final_cloud_colors.extend(curr_colors.tolist())

                crop_index = crop_index + 1

        if len(non_normalized_points) == 0:
            self.get_logger().info("No points in the cloud")
            return

        final_cloud_points = torch.cat(
            non_normalized_points, dim=0).cpu().detach().numpy()
        final_cloud.points = o3d.utility.Vector3dVector(
            np.asarray(final_cloud_points, dtype=np.float32))
        final_cloud.colors = o3d.utility.Vector3dVector(
            np.asarray(final_cloud_colors, dtype=np.float32))

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
