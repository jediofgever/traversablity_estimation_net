// Copyright (c) 2023 Fetullah Atas, Norwegian University of Life Sciences
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/publisher.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vision_msgs/msg/detection3_d_array.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree.h>
#include <Eigen/Dense>

namespace vox_nav_misc
{

class traversability_smoother : public rclcpp::Node
{
private:
  // tf buffer and listener
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  typedef message_filters::sync_policies::ApproximateTime<vision_msgs::msg::Detection3DArray,
                                                          sensor_msgs::msg::PointCloud2>
      BoxLidarpprxTimeSyncPolicy;
  typedef message_filters::Synchronizer<BoxLidarpprxTimeSyncPolicy> BoxLidarApprxTimeSyncer;

  message_filters::Subscriber<vision_msgs::msg::Detection3DArray> detection_boxes_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_subscriber_;

  std::shared_ptr<BoxLidarApprxTimeSyncer> time_syncher_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr smmothed_traversability_pub_;

  float kdtree_search_radius_;
  float cropped_cloud_voxel_size_;

public:
  traversability_smoother();
  ~traversability_smoother();

  /**
   * @brief ousterCamCallback
   * @param boc
   * @param cloud
   * This is the callback function for the synchronized lidar and camera messages
   */
  void boxLidarCallback(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& boxes,
                        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud);

  std::tuple<double, double, double> scalar_to_rgb(double scalar)
  {
    // Ensure scalar is within the range [0, 1]
    if (scalar < 0.0)
      scalar = 0.0;
    if (scalar > 1.0)
      scalar = 1.0;

    // Map the scalar value directly to the HSV color space
    double hue = (1.0 - scalar) * 240.0;  // 0 to 1 maps to 240° to 0° (blue to red)
    double saturation = 1.0;              // Full saturation
    double value = 1.0;                   // Full brightness

    // Convert HSV to RGB
    hue /= 60.0;
    double chroma = value * saturation;
    double x = chroma * (1.0 - std::abs(std::fmod(hue, 2.0) - 1.0));

    double r, g, b;
    if (0.0 <= hue && hue < 1.0)
    {
      r = chroma;
      g = x;
      b = 0.0;
    }
    else if (1.0 <= hue && hue < 2.0)
    {
      r = x;
      g = chroma;
      b = 0.0;
    }
    else if (2.0 <= hue && hue < 3.0)
    {
      r = 0.0;
      g = chroma;
      b = x;
    }
    else if (3.0 <= hue && hue < 4.0)
    {
      r = 0.0;
      g = x;
      b = chroma;
    }
    else if (4.0 <= hue && hue < 5.0)
    {
      r = x;
      g = 0.0;
      b = chroma;
    }
    else
    {
      r = chroma;
      g = 0.0;
      b = x;
    }

    double m = value - chroma;
    r += m;
    g += m;
    b += m;

    return std::make_tuple(r, g, b);
  }

  const double EPSILON = std::numeric_limits<double>::epsilon();

  std::tuple<int, int, int> convert_to_rgb(double minval, double maxval, double val,
                                           const std::vector<std::tuple<int, int, int>>& colors)
  {
    // Determine where the given value falls proportionality within
    // the range from minval->maxval and scale that fractional value
    // by the total number in the `colors` palette.
    double i_f = (val - minval) / (maxval - minval) * (colors.size() - 1);

    // Determine the lower index of the pair of color indices this
    // value corresponds and its fractional distance between the lower
    // and the upper colors.
    int i = static_cast<int>(i_f);
    double f = i_f - i;

    // Does it fall exactly on one of the color points?
    if (f < EPSILON)
    {
      return colors[i];
    }
    else
    {
      // Return a color linearly interpolated between the range of it and the following one.
      int r1, g1, b1;
      std::tie(r1, g1, b1) = colors[i];

      int r2, g2, b2;
      std::tie(r2, g2, b2) = colors[i + 1];

      return std::make_tuple(static_cast<int>(r1 + f * (r2 - r1)), static_cast<int>(g1 + f * (g2 - g1)),
                             static_cast<int>(b1 + f * (b2 - b1)));
    }
  }
};

traversability_smoother::traversability_smoother() : rclcpp::Node("traversability_smoother_rclcpp_node")
{
  // init tf buffer and listener
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  detection_boxes_subscriber_.subscribe(this, "boxes", rmw_qos_profile_sensor_data);
  lidar_subscriber_.subscribe(this, "points", rmw_qos_profile_sensor_data);

  time_syncher_.reset(
      new BoxLidarApprxTimeSyncer(BoxLidarpprxTimeSyncPolicy(50), detection_boxes_subscriber_, lidar_subscriber_));
  time_syncher_->registerCallback(
      std::bind(&traversability_smoother::boxLidarCallback, this, std::placeholders::_1, std::placeholders::_2));

  smmothed_traversability_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("smoothed_traversability", 1);

  declare_parameter("kdtree_search_radius", 0.3);
  get_parameter("kdtree_search_radius", kdtree_search_radius_);
  declare_parameter("downsample_voxel_size", 0.1);
  get_parameter("cropped_cloud_voxel_size", cropped_cloud_voxel_size_);

  // inform user the node has started
  RCLCPP_INFO(get_logger(), "traversability_smoother_rclcpp_node has started.");
}

traversability_smoother::~traversability_smoother()
{
  // inform user the node has shutdown
  RCLCPP_INFO(get_logger(), "traversability_smoother_rclcpp_node has shutdown.");
}

void traversability_smoother::boxLidarCallback(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& boxes,
                                               const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud)
{
  RCLCPP_INFO(get_logger(), "Transforming pointcloud to image");

  // log number of boxes
  RCLCPP_INFO(get_logger(), "Number of boxes: %d", boxes->detections.size());

  // convert to pcl pointcloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::fromROSMsg(*cloud, *pcl_cloud);

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_stacked(new pcl::PointCloud<pcl::PointXYZI>);

  // loop through all boxes and crop pointcloud
  for (auto box : boxes->detections)
  {
    // get box center
    auto box_center = box.bbox.center;
    // get box dimensions
    auto box_dims = box.bbox.size;
    // hypothesis
    auto hypothesis = box.results[0].hypothesis;

    Eigen::Vector4f min_point;
    Eigen::Vector4f max_point;
    min_point[0] = box_center.position.x - box_dims.x / 2.0;
    min_point[1] = box_center.position.y - box_dims.y / 2.0;
    min_point[2] = box_center.position.z - box_dims.z / 2.0;
    min_point[3] = 1.0;
    max_point[0] = box_center.position.x + box_dims.x / 2.0;
    max_point[1] = box_center.position.y + box_dims.y / 2.0;
    max_point[2] = box_center.position.z + box_dims.z / 2.0;
    max_point[3] = 1.0;

    // crop pointcloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropped_pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::CropBox<pcl::PointXYZRGB> crop_box_filter;
    crop_box_filter.setInputCloud(pcl_cloud);
    crop_box_filter.setMin(min_point);
    crop_box_filter.setMax(max_point);
    crop_box_filter.filter(*cropped_pcl_cloud);

    for (auto point : cropped_pcl_cloud->points)
    {
      pcl::PointXYZI point_i;
      point_i.x = point.x;
      point_i.y = point.y;
      point_i.z = point.z;
      point_i.intensity = hypothesis.score;
      pcl_cloud_stacked->points.push_back(point_i);
    }
  }

  // Now create an Octree from the point cloud
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> octree(0.2);
  octree.setInputCloud(pcl_cloud_stacked);
  octree.addPointsFromInputCloud();

  // iterate through all voxels in the octree

  // get occupied voxel centers
  std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI>> occupied_voxel_centers;
  octree.getOccupiedVoxelCenters(occupied_voxel_centers);

  for (auto voxel_center : occupied_voxel_centers)
  {
    // get all points in a sphere around this voxel center
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = 0.3;
    if (octree.radiusSearch(voxel_center, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
    {
      float mean_intensity = 0.0;
      for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
      {
        mean_intensity += pcl_cloud_stacked->points[pointIdxRadiusSearch[i]].intensity;
      }
      mean_intensity /= pointIdxRadiusSearch.size();

      for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
      {
        pcl_cloud_stacked->points[pointIdxRadiusSearch[i]].intensity = mean_intensity;
      }
    }
  }

  // downsample stacked pointcloud to reduce computation
  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_pcl_cloud_stacked(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  voxel_grid_filter.setInputCloud(pcl_cloud_stacked);
  voxel_grid_filter.setLeafSize(0.05, 0.05, 0.05);
  voxel_grid_filter.filter(*downsampled_pcl_cloud_stacked);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

  float min = 0.0;
  float max = 1.0;
  std::vector<std::tuple<int, int, int>> colors;
  colors.push_back(std::make_tuple(0, 0, 255));  // blue
  colors.push_back(std::make_tuple(0, 255, 0));  // green
  colors.push_back(std::make_tuple(255, 0, 0));  // red

  for (auto point : downsampled_pcl_cloud_stacked->points)
  {
    pcl::PointXYZRGB this_point;
    this_point.x = point.x;
    this_point.y = point.y;
    this_point.z = point.z;

    auto rgb = convert_to_rgb(min, max, point.intensity, colors);
    this_point.r = std::get<0>(rgb);
    this_point.g = std::get<1>(rgb);
    this_point.b = std::get<2>(rgb);

    pcl_cloud_rgb->points.push_back(this_point);
  }

  // publish pointcloud
  sensor_msgs::msg::PointCloud2::SharedPtr pcl_cloud_rgb_msg(new sensor_msgs::msg::PointCloud2);
  pcl::toROSMsg(*pcl_cloud_rgb, *pcl_cloud_rgb_msg);
  pcl_cloud_rgb_msg->header.frame_id = cloud->header.frame_id;
  pcl_cloud_rgb_msg->header.stamp = cloud->header.stamp;
  smmothed_traversability_pub_->publish(*pcl_cloud_rgb_msg);
}

}  // namespace vox_nav_misc

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vox_nav_misc::traversability_smoother>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
