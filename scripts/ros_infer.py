
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
from sensor_msgs_py.point_cloud2 import read_points_numpy
from std_msgs.msg import Header
import sensor_msgs.msg as sensor_msgs
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from dataset import pc_normalize
from pointnet import PointNet


class PCDSubPubNode(Node):

    def __init__(self):
        super().__init__('pcd_subsriber_node')
        # Set up a subscription to the 'pcd' topic with a callback to the
        # function `listener_callback`

        self.topic_name = 'map_cloud'
        self.topic_name = '/lio_sam/mapping/map_local'
        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,                            # Msg type
            self.topic_name,                                        # topic
            self.listener_callback,                             # Function to call
            qos_profile=rclpy.qos.qos_profile_sensor_data       # QoS
        )

        self.obstacle_pcd_publisher = self.create_publisher(
            sensor_msgs.PointCloud2,
            'traversable_map_cloud',
            qos_profile=rclpy.qos.qos_profile_sensor_data)

        self.box_publisher = self.create_publisher(
            Detection3DArray,
            'boxes',
            qos_profile=rclpy.qos.qos_profile_sensor_data)

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # set use_sim_time to true to use the simulation clock
        self.use_sim_time = True
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, self.use_sim_time)])
        self.net = PointNet()

        # load the model
        self.net.load_state_dict(torch.load("weights/epoch_290.pt"))
        self.net.eval()
        self.net.cuda()
        self.net.zero_grad()
        self.get_logger().info("Model loaded")

    def listener_callback(self, msg: sensor_msgs.PointCloud2):

        try:

            numpy_array_points = read_points_numpy(
                msg, field_names=("x", "y", "z"))

            from_frame_rel = 'map'
            to_frame_rel = 'base_link'

            trans = self.buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time())

            # Now create a open3d point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(numpy_array_points)

            # downsample the cloud
            # pcd = pcd.voxel_down_sample(voxel_size=0.05)

            self.infer(pcd, trans)

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def populate_and_publish_boxes(self,  boxes, header):
        # populate the boxes
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

    def infer(self, pcd: o3d.geometry.PointCloud, trans: TransformStamped):

        # print the number of points in the cloud

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

        # crop the cloud to the region of interest
        min_corner = [-10, -10, -3.5]
        max_corner = [10, 10, 3.5]
        x_step_size = 1
        y_step_size = 0.5

        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_corner, max_corner))

        print("Number of points in the cloud", len(pcd.points))

        x_dist = abs(max_corner[0] - min_corner[0])
        y_dist = abs(max_corner[1] - min_corner[1])

        geometries = []
        boxes = []
        data_list = []
        non_normalized_points = []

        for x in range(0, int(x_dist / x_step_size+1)):

            for y in range(0, int(y_dist / y_step_size+1)):

                current_min_corner = [
                    min_corner[0] + x_step_size * x,
                    min_corner[1] + y_step_size * y,
                    -3.5,
                ]

                current_max_corner = [
                    current_min_corner[0] + x_step_size,
                    current_min_corner[1] + y_step_size,
                    3.5,
                ]

                this_box = o3d.geometry.AxisAlignedBoundingBox(
                    current_min_corner, current_max_corner
                )

                boxes.append(this_box)

                cropped_pcd = pcd.crop(this_box)

                if len(cropped_pcd.points) < 10:
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
                        deep_copy = copy.deepcopy(cropped_pcd)
                        points = np.asarray(
                            deep_copy.points).astype(np.float32)
                        points = pc_normalize(points)
                        deep_copy.points = o3d.utility.Vector3dVector(points)
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

        header = Header()
        header.frame_id = "base_link"
        header.stamp = self.get_clock().now().to_msg()

        self.populate_and_publish_boxes(boxes, header)

        batch_size = 128
        crop_index = 0
        final_cloud = o3d.geometry.PointCloud()
        final_cloud_points = []
        final_cloud_colors = []

        for x in self.batch(data_list, batch_size):
            # create a batch
            # empty torch tensor of size (batch_size, 3, 1)
            batches = []
            batch_index = 0
            for sample in x:
                batches.append(torch.ones((sample.size(dim=0)),
                               dtype=torch.long) * batch_index)
                batch_index = batch_index + 1

            batches = torch.cat(batches, dim=0).cuda()
            points = torch.cat(x, dim=0).cuda()

            predictions = self.net(points, batches).cpu().detach().numpy()

            # Bottlenek, too slow
            for p in predictions:
                p = np.clip(p, 0, 1)
                if p[0] < 0.2:
                    geometries[crop_index].paint_uniform_color(
                        [0, 1 - p[0], 0])
                else:
                    geometries[crop_index].paint_uniform_color([p[0], 0, 0])

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

        final_points = np.asarray(final_cloud.points, dtype=np.float32)
        final_colors = np.asarray(final_cloud.colors, dtype=np.float32)

        # concat the points and colors to create N X 6 matrix
        points_colors = np.concatenate((final_points, final_colors), axis=1)

        # add additional alpha channel to the colors to create N X 7 matrix
        points_colors = np.concatenate((points_colors, np.ones(
            (points_colors.shape[0], 1), dtype=np.float32)), axis=1)

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        data = points_colors.astype(dtype).tobytes()

        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyzrgba')]

        final_ros_cloud = sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points_colors.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 7),
            row_step=(itemsize * 7 * points_colors.shape[0]),
            data=data
        )

        print("Inferred in seconds", time.time() - start)

        # according to this prediction we can color the cloud

        self.obstacle_pcd_publisher.publish(final_ros_cloud)


def main():
    rclpy.init()
    node = PCDSubPubNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
