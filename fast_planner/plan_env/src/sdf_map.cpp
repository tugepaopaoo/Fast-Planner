/**
* This file is part of Fast-Planner.
*
* Copyright 2019 Boyu Zhou, Aerial Robotics Group, Hong Kong University of Science and Technology, <uav.ust.hk>
* Developed by Boyu Zhou <bzhouai at connect dot ust dot hk>, <uv dot boyuzhou at gmail dot com>
* for more information see <https://github.com/HKUST-Aerial-Robotics/Fast-Planner>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* Fast-Planner is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Fast-Planner is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with Fast-Planner. If not, see <http://www.gnu.org/licenses/>.
*/



#include "plan_env/sdf_map.h"

// #define current_img_ md_.depth_image_[image_cnt_ & 1]
// #define last_img_ md_.depth_image_[!(image_cnt_ & 1)]

void SDFMap::initMap(ros::NodeHandle& nh) {
  node_ = nh;

  /* get parameter */
  double x_size, y_size, z_size;
  node_.param("sdf_map/resolution", mp_.resolution_, -1.0);
  node_.param("sdf_map/map_size_x", x_size, -1.0);
  node_.param("sdf_map/map_size_y", y_size, -1.0);
  node_.param("sdf_map/map_size_z", z_size, -1.0);
  node_.param("sdf_map/local_update_range_x", mp_.local_update_range_(0), -1.0);
  node_.param("sdf_map/local_update_range_y", mp_.local_update_range_(1), -1.0);
  node_.param("sdf_map/local_update_range_z", mp_.local_update_range_(2), -1.0);
  node_.param("sdf_map/obstacles_inflation", mp_.obstacles_inflation_, -1.0);

  node_.param("sdf_map/fx", mp_.fx_, -1.0);
  node_.param("sdf_map/fy", mp_.fy_, -1.0);
  node_.param("sdf_map/cx", mp_.cx_, -1.0);
  node_.param("sdf_map/cy", mp_.cy_, -1.0);

  node_.param("sdf_map/use_depth_filter", mp_.use_depth_filter_, true);
  node_.param("sdf_map/depth_filter_tolerance", mp_.depth_filter_tolerance_, -1.0);
  node_.param("sdf_map/depth_filter_maxdist", mp_.depth_filter_maxdist_, -1.0);
  node_.param("sdf_map/depth_filter_mindist", mp_.depth_filter_mindist_, -1.0);
  node_.param("sdf_map/depth_filter_margin", mp_.depth_filter_margin_, -1);
  node_.param("sdf_map/k_depth_scaling_factor", mp_.k_depth_scaling_factor_, -1.0);
  node_.param("sdf_map/skip_pixel", mp_.skip_pixel_, -1);   //为了提高地图融合效率，进行下采样；若图像分辨率较低，设置数值为1，禁止下采样

  node_.param("sdf_map/p_hit", mp_.p_hit_, 0.70);
  node_.param("sdf_map/p_miss", mp_.p_miss_, 0.35);
  node_.param("sdf_map/p_min", mp_.p_min_, 0.12);
  node_.param("sdf_map/p_max", mp_.p_max_, 0.97);
  node_.param("sdf_map/p_occ", mp_.p_occ_, 0.80);
  node_.param("sdf_map/min_ray_length", mp_.min_ray_length_, -0.1);
  node_.param("sdf_map/max_ray_length", mp_.max_ray_length_, -0.1);

  node_.param("sdf_map/esdf_slice_height", mp_.esdf_slice_height_, -0.1);
  node_.param("sdf_map/visualization_truncate_height", mp_.visualization_truncate_height_, -0.1);
  node_.param("sdf_map/virtual_ceil_height", mp_.virtual_ceil_height_, -0.1);

  node_.param("sdf_map/show_occ_time", mp_.show_occ_time_, false);
  node_.param("sdf_map/show_esdf_time", mp_.show_esdf_time_, false);
  node_.param("sdf_map/pose_type", mp_.pose_type_, 1);

  node_.param("sdf_map/frame_id", mp_.frame_id_, string("world"));
  node_.param("sdf_map/local_bound_inflate", mp_.local_bound_inflate_, 1.0);  
  node_.param("sdf_map/local_map_margin", mp_.local_map_margin_, 1);
  node_.param("sdf_map/ground_height", mp_.ground_height_, 1.0);

  // 局部边界膨胀值,膨胀后才能包含边界点膨胀后产生的点
  mp_.local_bound_inflate_ = max(mp_.resolution_, mp_.local_bound_inflate_);
  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_.ground_height_);   //地图原点在左下角
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  mp_.prob_hit_log_ = logit(mp_.p_hit_);
  mp_.prob_miss_log_ = logit(mp_.p_miss_);
  mp_.clamp_min_log_ = logit(mp_.p_min_);       // 设定的占据的概率最小值
  mp_.clamp_max_log_ = logit(mp_.p_max_);       // 设定的占据的概率最大值
  mp_.min_occupancy_log_ = logit(mp_.p_occ_);   // 最终能够判断占据的概率阈值
  mp_.unknown_flag_ = 0.01;

  cout << "hit: " << mp_.prob_hit_log_ << endl;
  cout << "miss: " << mp_.prob_miss_log_ << endl;
  cout << "min log: " << mp_.clamp_min_log_ << endl;
  cout << "max: " << mp_.clamp_max_log_ << endl;
  cout << "thresh log: " << mp_.min_occupancy_log_ << endl;

  for (int i = 0; i < 3; ++i) mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;

  mp_.map_min_idx_ = Eigen::Vector3i::Zero();
  mp_.map_max_idx_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();

  // initialize data buffers

  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);

  md_.occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
  md_.occupancy_buffer_neg = vector<char>(buffer_size, 0);
  md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);

  md_.distance_buffer_ = vector<double>(buffer_size, 10000);
  md_.distance_buffer_neg_ = vector<double>(buffer_size, 10000);
  md_.distance_buffer_all_ = vector<double>(buffer_size, 10000);

  md_.count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_.count_hit_ = vector<short>(buffer_size, 0);
  md_.flag_rayend_ = vector<char>(buffer_size, -1);
  md_.flag_traverse_ = vector<char>(buffer_size, -1);

  md_.tmp_buffer1_ = vector<double>(buffer_size, 0);
  md_.tmp_buffer2_ = vector<double>(buffer_size, 0);
  md_.raycast_num_ = 0;

  md_.proj_points_.resize(640 * 480 / mp_.skip_pixel_ / mp_.skip_pixel_);
  md_.proj_points_cnt = 0;

  /* init callback */

  depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node_, "/sdf_map/depth", 50));

  if (mp_.pose_type_ == POSE_STAMPED) {
    pose_sub_.reset(
        new message_filters::Subscriber<geometry_msgs::PoseStamped>(node_, "/sdf_map/pose", 25));

    sync_image_pose_.reset(new message_filters::Synchronizer<SyncPolicyImagePose>(
        SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
    sync_image_pose_->registerCallback(boost::bind(&SDFMap::depthPoseCallback, this, _1, _2));

  } else if (mp_.pose_type_ == ODOMETRY) {
    odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(node_, "/sdf_map/odom", 100));

    sync_image_odom_.reset(new message_filters::Synchronizer<SyncPolicyImageOdom>(
        SyncPolicyImageOdom(100), *depth_sub_, *odom_sub_));
    sync_image_odom_->registerCallback(boost::bind(&SDFMap::depthOdomCallback, this, _1, _2));
  }

  // use odometry and point cloud

  indep_cloud_sub_ =
      node_.subscribe<sensor_msgs::PointCloud2>("/sdf_map/cloud", 10, &SDFMap::cloudCallback, this);
  indep_odom_sub_ =
      node_.subscribe<nav_msgs::Odometry>("/sdf_map/odom", 10, &SDFMap::odomCallback, this);

  occ_timer_ = node_.createTimer(ros::Duration(0.05), &SDFMap::updateOccupancyCallback, this);
  esdf_timer_ = node_.createTimer(ros::Duration(0.05), &SDFMap::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.05), &SDFMap::visCallback, this);

  map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy", 10);
  map_inf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_inflate", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/sdf_map/update_range", 10);

  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  depth_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/depth_cloud", 10);

  md_.occ_need_update_ = false;
  md_.local_updated_ = false;
  md_.esdf_need_update_ = false;
  md_.has_first_depth_ = false;
  md_.has_odom_ = false;
  md_.has_cloud_ = false;
  md_.image_cnt_ = 0;

  md_.esdf_time_ = 0.0;
  md_.fuse_time_ = 0.0;
  md_.update_num_ = 0;
  md_.max_esdf_time_ = 0.0;
  md_.max_fuse_time_ = 0.0;

  rand_noise_ = uniform_real_distribution<double>(-0.2, 0.2);
  rand_noise2_ = normal_distribution<double>(0, 0.2);
  random_device rd;
  eng_ = default_random_engine(rd());
}

void SDFMap::resetBuffer() {
  Eigen::Vector3d min_pos = mp_.map_min_boundary_;
  Eigen::Vector3d max_pos = mp_.map_max_boundary_;

  resetBuffer(min_pos, max_pos);

  md_.local_bound_min_ = Eigen::Vector3i::Zero();
  md_.local_bound_max_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
}

void SDFMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos) {

  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);

  boundIndex(min_id);
  boundIndex(max_id);

  /* reset occ and dist buffer */
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
        md_.distance_buffer_[toAddress(x, y, z)] = 10000;
      }
}

//计算ESDF地图中的欧式距离数值（一维）,f_get_val和f_set_val是函数作为另一个函数的参数
template <typename F_get_val, typename F_set_val>
void SDFMap::fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim) {
  int v[mp_.map_voxel_num_(dim)];
  double z[mp_.map_voxel_num_(dim) + 1];

  int k = start;
  v[start] = start;    
  z[start] = -std::numeric_limits<double>::max();
  z[start + 1] = std::numeric_limits<double>::max();  //返回编译器允许的 double型数 的最大值

  for (int q = start + 1; q <= end; q++) {
    k++;
    double s;

    do {
      k--;
      s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
    } while (s <= z[k]);

    k++;

    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::max();
  }

  k = start;

  for (int q = start; q <= end; q++) {
    while (z[k + 1] < q) k++;
    double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
    f_set_val(q, val);
  }
}

void SDFMap::updateESDF3d() {
  Eigen::Vector3i min_esdf = md_.local_bound_min_;
  Eigen::Vector3i max_esdf = md_.local_bound_max_;

  /* ========== compute positive DT ========== */

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
      fillESDF(
          [&](int z) {                                                   // lambda 匿名函数
            return md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 1 ?
                0 :
                std::numeric_limits<double>::max();
          },
          [&](int z, double val) { md_.tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
          max_esdf[2], 2);
    }
  }

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int y) { return md_.tmp_buffer1_[toAddress(x, y, z)]; }, //把上一维的计算结果tmp_buffer1_，作为这一维度计算的基础
               [&](int y, double val) { md_.tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
               max_esdf[1], 1);
    }
  }

  for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int x) { return md_.tmp_buffer2_[toAddress(x, y, z)]; },
               [&](int x, double val) {
                 //val中的值缺少真实物理尺寸，乘上resolution_后才有真实的地图尺度信息
                 //前面计算的距离都是欧式距离的平方，最后一个维度计算完并开方才是实际的距离
                 //因为开方这一计算较为耗时，为方便只在最后进行一次
                 md_.distance_buffer_[toAddress(x, y, z)] = mp_.resolution_ * std::sqrt(val);
                 //  min(mp_.resolution_ * std::sqrt(val),
                 //      md_.distance_buffer_[toAddress(x, y, z)]);
               },
               min_esdf[0], max_esdf[0], 0);
    }
  }

  //将空闲栅格和占据栅格的标志符号交换，以便于用同一套程序计算，障碍物内部的栅格esdf距离，最后计算出距离要都加上负号
  /* ========== compute negative distance ========== */
  for (int x = min_esdf(0); x <= max_esdf(0); ++x)          
    for (int y = min_esdf(1); y <= max_esdf(1); ++y)
      for (int z = min_esdf(2); z <= max_esdf(2); ++z) {

        int idx = toAddress(x, y, z);
        if (md_.occupancy_buffer_inflate_[idx] == 0) {
          md_.occupancy_buffer_neg[idx] = 1;

        } else if (md_.occupancy_buffer_inflate_[idx] == 1) {
          md_.occupancy_buffer_neg[idx] = 0;
        } else {
          ROS_ERROR("what?");
        }
      }

  ros::Time t1, t2;

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
      fillESDF(
          [&](int z) {
            return md_.occupancy_buffer_neg[x * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) +
                                            y * mp_.map_voxel_num_(2) + z] == 1 ?
                0 :
                std::numeric_limits<double>::max();
          },
          [&](int z, double val) { md_.tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
          max_esdf[2], 2);
    }
  }

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int y) { return md_.tmp_buffer1_[toAddress(x, y, z)]; },
               [&](int y, double val) { md_.tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
               max_esdf[1], 1);
    }
  }

  for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int x) { return md_.tmp_buffer2_[toAddress(x, y, z)]; },
               [&](int x, double val) {
                 md_.distance_buffer_neg_[toAddress(x, y, z)] = mp_.resolution_ * std::sqrt(val);
               },
               min_esdf[0], max_esdf[0], 0);
    }
  }


  //上述两个地图值融合起来，就是整个esdf地图的距离值，占据栅格外部的距离和占据栅格内部的距离
  /* ========== combine pos and neg DT ========== */          
  for (int x = min_esdf(0); x <= max_esdf(0); ++x)
    for (int y = min_esdf(1); y <= max_esdf(1); ++y)
      for (int z = min_esdf(2); z <= max_esdf(2); ++z) {

        int idx = toAddress(x, y, z);
        md_.distance_buffer_all_[idx] = md_.distance_buffer_[idx];

        if (md_.distance_buffer_neg_[idx] > 0.0)
          //distance_buffer_neg_需要取负号，才能表示这是障碍物内部的距离
          //障碍物边界上的距离为0，所以(-md_.distance_buffer_neg_[idx] + mp_.resolution_)这里有了这样的处理
          md_.distance_buffer_all_[idx] += (-md_.distance_buffer_neg_[idx] + mp_.resolution_); 
      }
}


//如果occ不为1或0则返回INVALID_IDX ，否则返回pos在地图中的地址。若occ==1 ，将该点为障碍物的概率上升 。 
int SDFMap::setCacheOccupancy(Eigen::Vector3d pos, int occ) {
  if (occ != 1 && occ != 0) return INVALID_IDX;

  Eigen::Vector3i id;
  posToIndex(pos, id);
  int idx_ctns = toAddress(id);    // 将三维坐标索引变为一维栅格坐标索引。idx_ctns 可能是表示 "index in continuous" 的缩写

  md_.count_hit_and_miss_[idx_ctns] += 1;     //每被经过1次，就+1；不管是占据还是空闲，都会+1

  if (md_.count_hit_and_miss_[idx_ctns] == 1) {  //如果该体素第一次被发现，就把他加入cache_voxel_数组中，用于缓存地图数据，提高访问效率。
    md_.cache_voxel_.push(id);
  }

  if (occ == 1) md_.count_hit_[idx_ctns] += 1;   //如果是占据，count_hit_的值+1。即增加了该体素被占据的概率。

  return idx_ctns;
}


/* 通过相机投影模型，将图像坐标系上的深度点云投影⾄相机坐标系，再通过相机的位姿将相机坐标系上点云投影⾄世界坐标
系，最后将所有点存⼊md.proj_points_这⼀vector容器中。*/
void SDFMap::projectDepthImage() {
  // md_.proj_points_.clear();
  md_.proj_points_cnt = 0;

  uint16_t* row_ptr;
  // int cols = current_img_.cols, rows = current_img_.rows;
  int cols = md_.depth_image_.cols;
  int rows = md_.depth_image_.rows;

  double depth;

  Eigen::Matrix3d camera_r = md_.camera_q_.toRotationMatrix();

  // cout << "rotate: " << md_.camera_q_.toRotationMatrix() << endl;
  // std::cout << "pos in proj: " << md_.camera_pos_ << std::endl;

  if (!mp_.use_depth_filter_) {
    for (int v = 0; v < rows; v++) {
      row_ptr = md_.depth_image_.ptr<uint16_t>(v);

      for (int u = 0; u < cols; u++) {

        Eigen::Vector3d proj_pt;
        depth = (*row_ptr++) / mp_.k_depth_scaling_factor_;
        proj_pt(0) = (u - mp_.cx_) * depth / mp_.fx_;
        proj_pt(1) = (v - mp_.cy_) * depth / mp_.fy_;
        proj_pt(2) = depth;

        proj_pt = camera_r * proj_pt + md_.camera_pos_;

        if (u == 320 && v == 240) std::cout << "depth: " << depth << std::endl;
        md_.proj_points_[md_.proj_points_cnt++] = proj_pt;
      }
    }
  }
  /* use depth filter */
  else {

    if (!md_.has_first_depth_)
      md_.has_first_depth_ = true;
    else {
      Eigen::Vector3d pt_cur, pt_world, pt_reproj;

      Eigen::Matrix3d last_camera_r_inv;
      last_camera_r_inv = md_.last_camera_q_.inverse();
      const double inv_factor = 1.0 / mp_.k_depth_scaling_factor_;

      for (int v = mp_.depth_filter_margin_; v < rows - mp_.depth_filter_margin_; v += mp_.skip_pixel_) {
        row_ptr = md_.depth_image_.ptr<uint16_t>(v) + mp_.depth_filter_margin_;

        for (int u = mp_.depth_filter_margin_; u < cols - mp_.depth_filter_margin_;
             u += mp_.skip_pixel_) {

          depth = (*row_ptr) * inv_factor;
          row_ptr = row_ptr + mp_.skip_pixel_;

          // filter depth
          // depth += rand_noise_(eng_);
          // if (depth > 0.01) depth += rand_noise2_(eng_);

          if (*row_ptr == 0) {
            depth = mp_.max_ray_length_ + 0.1;
          } else if (depth < mp_.depth_filter_mindist_) {
            continue;
          } else if (depth > mp_.depth_filter_maxdist_) {
            depth = mp_.max_ray_length_ + 0.1;
          }

          // project to world frame
          pt_cur(0) = (u - mp_.cx_) * depth / mp_.fx_;
          pt_cur(1) = (v - mp_.cy_) * depth / mp_.fy_;
          pt_cur(2) = depth;

          pt_world = camera_r * pt_cur + md_.camera_pos_;
          // if (!isInMap(pt_world)) {
          //   pt_world = closetPointInMap(pt_world, md_.camera_pos_);
          // }

          md_.proj_points_[md_.proj_points_cnt++] = pt_world;

          // check consistency with last image, disabled...
          if (false) {
            pt_reproj = last_camera_r_inv * (pt_world - md_.last_camera_pos_);
            double uu = pt_reproj.x() * mp_.fx_ / pt_reproj.z() + mp_.cx_;
            double vv = pt_reproj.y() * mp_.fy_ / pt_reproj.z() + mp_.cy_;

            if (uu >= 0 && uu < cols && vv >= 0 && vv < rows) {
              if (fabs(md_.last_depth_image_.at<uint16_t>((int)vv, (int)uu) * inv_factor -
                       pt_reproj.z()) < mp_.depth_filter_tolerance_) {
                md_.proj_points_[md_.proj_points_cnt++] = pt_world;
              }
            } else {
              md_.proj_points_[md_.proj_points_cnt++] = pt_world;
            }
          }
        }
      }
    }
  }

  /* maintain camera pose for consistency check */

  md_.last_camera_pos_ = md_.camera_pos_;
  md_.last_camera_q_ = md_.camera_q_;
  md_.last_depth_image_ = md_.depth_image_;
}



/* 若projectDepthImage()中添加了新的点云，则对于每个新加入的点判断是否已经在地图中。
当该点不在地图范围内，则将该点挪到地图边界上，如果该点到相机的距离大于最远可测距离则把点挪到最远可测距离上，最后得到该点在地图中的地址；
当该点在地图范围内，若该点到相机的距离大于最远可测距离，则把点挪到最远可测距离上并得到该点在地图中的地址，
否则在得到该点在地图中的地址的基础上,将该点为障碍物概率增加，将该点的md_.flag_rayend_设为当前调用raycastProcess()的次数。
更新边界，并将md_.local_updated_置为true 。最后根据概率更新栅格地图。
*/
void SDFMap::raycastProcess() {
  // if (md_.proj_points_.size() == 0)
  if (md_.proj_points_cnt == 0) return;

  ros::Time t1, t2;

  // 初始化时，md_.raycast_num_ = 0
  md_.raycast_num_ += 1;

  int vox_idx;
  double length;

  // bounding box of updated region  更新区域的边界
  // 把最大值赋给min_x是为了后续方便计算更新区域的边界，该操作类似与函数cloudCallback()中操作相同
  double min_x = mp_.map_max_boundary_(0);
  double min_y = mp_.map_max_boundary_(1);
  double min_z = mp_.map_max_boundary_(2);

  double max_x = mp_.map_min_boundary_(0);
  double max_y = mp_.map_min_boundary_(1);
  double max_z = mp_.map_min_boundary_(2);

  RayCaster raycaster;
  Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
  Eigen::Vector3d ray_pt, pt_w;

  for (int i = 0; i < md_.proj_points_cnt; ++i) {
    pt_w = md_.proj_points_[i];

    // set flag for projected point

    // 判断该点是否在地图范围内
    // 若不在地图范围内，在地图范围内找一个与该点最近的点；
    // 如果该点到相机的距离大于最远可测距离,则把点挪到最远可测距离上，并设为空闲
    if (!isInMap(pt_w)) {
      pt_w = closetPointInMap(pt_w, md_.camera_pos_);     // 在地图范围内找一个最近点（地图边界上的点）

      length = (pt_w - md_.camera_pos_).norm();
      if (length > mp_.max_ray_length_) {
        pt_w = (pt_w - md_.camera_pos_) / length * mp_.max_ray_length_ + md_.camera_pos_;
      }
      vox_idx = setCacheOccupancy(pt_w, 0);   //设置为空闲

    } else {             
      // 若该点在地图内:1.不在射线追踪范围,则把点挪到最远可测距离上,设为空闲；2.在射线追踪范围,设为占据
      length = (pt_w - md_.camera_pos_).norm();

      if (length > mp_.max_ray_length_) {
        pt_w = (pt_w - md_.camera_pos_) / length * mp_.max_ray_length_ + md_.camera_pos_;
        vox_idx = setCacheOccupancy(pt_w, 0);
      } else {
        vox_idx = setCacheOccupancy(pt_w, 1); 
      }
    }

    // 确定地图更新区域边界：获取点云区域的最大坐标，即地图更新区域边界的最大坐标
    max_x = max(max_x, pt_w(0));
    max_y = max(max_y, pt_w(1));
    max_z = max(max_z, pt_w(2));

    // 确定地图更新区域边界：获取点云区域的最小坐标，即地图更新区域边界的最小坐标
    min_x = min(min_x, pt_w(0));
    min_y = min(min_y, pt_w(1));
    min_z = min(min_z, pt_w(2));

    // raycasting between camera center and point
    // vox_idx是表示当前体素的索引坐标
    // enum { POSE_STAMPED = 1, ODOMETRY = 2, INVALID_IDX = -10000 };
    // 在setCacheOccupancy函数中有 /* if (occ != 1 && occ != 0)  return INVALID_IDX; */
    // vox_idx 表示当前处理的体素的栅格索引，如果这个体素已经被处理过，就没有必要再次处理，提高效率
    if (vox_idx != INVALID_IDX) {
      if (md_.flag_rayend_[vox_idx] == md_.raycast_num_) {
        continue;
      } else {
        md_.flag_rayend_[vox_idx] = md_.raycast_num_;
      }
    }

    // pt_w / mp_.resolution_表示点云的栅格索引坐标, md_.camera_pos_ / mp_.resolution_表示相机的栅格索引坐标
    // raycaster.setInput的具体实现见raycast.cpp文件
    raycaster.setInput(pt_w / mp_.resolution_, md_.camera_pos_ / mp_.resolution_);

    // 通过raycast.step函数从射线终点开始向相机点步进。并且将每一个中途点都利用setCacheOccupancy函数置一次free。
    // 需要注意的是，每一个中途点还利用md_.flag_traverse_容器进行了判断，如果对应序列处的值不是本轮raycast的num,则将其置为本轮的racastnum.
    // 否则说明这一点及之后的点都已经被raycast过了，因此跳出当前射线终点的raycast循环。
    // raycaster.step(ray_pt)=false表示射线追踪到了最后一个点

    // 通过调用 raycaster.step(ray_pt) 来获取当前射线追踪点的栅格坐标ray_pt。
    // 然后通过将射线追踪点的坐标转换为位置坐标 tmp，计算出当前射线长度 length。
    while (raycaster.step(ray_pt)) {
      Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;  // half偏移的向量,加上后表示栅格中心位置。
      length = (tmp - md_.camera_pos_).norm();

      // if (length < mp_.min_ray_length_) break;

      vox_idx = setCacheOccupancy(tmp, 0);

      if (vox_idx != INVALID_IDX) {
        if (md_.flag_traverse_[vox_idx] == md_.raycast_num_) {
          break;
        } else {
          md_.flag_traverse_[vox_idx] = md_.raycast_num_;
        }
      }
    }
  }

  // determine the local bounding box for updating ESDF
  // 比较点云的最小位置 和 相机的位置, 取二者最小值
  min_x = min(min_x, md_.camera_pos_(0));
  min_y = min(min_y, md_.camera_pos_(1));
  min_z = min(min_z, md_.camera_pos_(2));
  
  // 比较点云的最大位置 和 相机的位置, 取二者最大值
  max_x = max(max_x, md_.camera_pos_(0));
  max_y = max(max_y, md_.camera_pos_(1));
  max_z = max(max_z, md_.camera_pos_(2));
  max_z = max(max_z, mp_.ground_height_);

  // 将位置坐标转换为栅格坐标,重置更新栅格的范围
  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_.local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_.local_bound_min_);

  // 确保待更新栅格均在地图内.若不在,取地图边界值予以替代
  // 包含esdf_inf扩展后，才能包含边界上的点膨胀后产生的esdf地图
  int esdf_inf = ceil(mp_.local_bound_inflate_ / mp_.resolution_);
  md_.local_bound_max_ += esdf_inf * Eigen::Vector3i(1, 1, 0);
  md_.local_bound_min_ -= esdf_inf * Eigen::Vector3i(1, 1, 0);
  boundIndex(md_.local_bound_min_);
  boundIndex(md_.local_bound_max_);
  
  // 作用详见代码 if (md_.local_updated_) { clearAndInflateLocalMap(); }
  md_.local_updated_ = true;

  // update occupancy cached（已缓存） in queue
  Eigen::Vector3d local_range_min = md_.camera_pos_ - mp_.local_update_range_;
  Eigen::Vector3d local_range_max = md_.camera_pos_ + mp_.local_update_range_;

  Eigen::Vector3i min_id, max_id;
  posToIndex(local_range_min, min_id);
  posToIndex(local_range_max, max_id);
  boundIndex(min_id);
  boundIndex(max_id);

  // std::cout << "cache all: " << md_.cache_voxel_.size() << std::endl;


  /* 当完成md.proj_points容器中所有点的raycast循环后，开始对md_.cache_voxel中的点进行循环判断。
  首先根据md_.count_hit及md_.count_hit_and_miss中对应位置的值判断当前voxel为障碍物的概率。
  并且如果当前点的log_odds_update是prob_hit_log，且md_.occupancy_buffer_中对应位置的概率值还没有超过最大值
  或当前点的log_odds_update是prob_miss_log，且md_.occupancy_buffer_中对应位置的概率值还没有低于最小值。
  且当前点是在局部地图范围内，则更新md_.occupancy_buffer_中的概率值。 */
  while (!md_.cache_voxel_.empty()) {

    Eigen::Vector3i idx = md_.cache_voxel_.front();
    int idx_ctns = toAddress(idx); // 将三维坐标换为一维线性索引，这种索引方式通常被称为“连续存储”或“基于坐标的线性索引”。
    md_.cache_voxel_.pop();        // idx_ctns 可能是表示 "index in continuous" 的缩写。
    
    // 若hit次数大于miss次数,判断为占据体素,log_odds_update =mp_.prob_hit_log_
    // 若miss次数大于hit次数,判断为空闲体素,log_odds_update =mp_.prob_miss_log_
    double log_odds_update =
        md_.count_hit_[idx_ctns] >= md_.count_hit_and_miss_[idx_ctns] - md_.count_hit_[idx_ctns] ?
        mp_.prob_hit_log_ :
        mp_.prob_miss_log_;

    md_.count_hit_[idx_ctns] = md_.count_hit_and_miss_[idx_ctns] = 0; //清零hit和miss的计数，准备处理下一个体素
    
    // 在进行实际的占据概率更新前，需要进行一些判断性的处理，如检查是否满足更新条件，是否在局部更新范围内等。

    //如果判断得当前体素已经是占据状态，并且当前的 log_odds 值已经达到上限，则跳过该体素；
    //如果判断当前体素是空闲状态，并且当前的 log_odds 值已经达到下限，则将该体素的占据概率设定为最小值，并跳过该体素。
    if (log_odds_update >= 0 && md_.occupancy_buffer_[idx_ctns] >= mp_.clamp_max_log_) {
      continue;
    } 
    else if (log_odds_update <= 0 && md_.occupancy_buffer_[idx_ctns] <= mp_.clamp_min_log_) {
      md_.occupancy_buffer_[idx_ctns] = mp_.clamp_min_log_;
      continue;
    }

    bool in_local = idx(0) >= min_id(0) && idx(0) <= max_id(0) && idx(1) >= min_id(1) &&
        idx(1) <= max_id(1) && idx(2) >= min_id(2) && idx(2) <= max_id(2);
    
    // 不在局部更新范围,将体素设定为占据概率的最小值
    if (!in_local) {
      md_.occupancy_buffer_[idx_ctns] = mp_.clamp_min_log_;
    }
    
    // 在局部更新范围,求得正确的占据概率
    md_.occupancy_buffer_[idx_ctns] =
        std::min(std::max(md_.occupancy_buffer_[idx_ctns] + log_odds_update, mp_.clamp_min_log_),
                 mp_.clamp_max_log_);
  }
}

// 在地图范围内找一个最近点（地图边界上的点）。基本原理:利用相似三角形，将超出范围的点拉回到地图边界上
Eigen::Vector3d SDFMap::closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt) {
  Eigen::Vector3d diff = pt - camera_pt;
  Eigen::Vector3d max_tc = mp_.map_max_boundary_ - camera_pt;
  Eigen::Vector3d min_tc = mp_.map_min_boundary_ - camera_pt;

  double min_t = 1000000;

  for (int i = 0; i < 3; ++i) {
    if (fabs(diff[i]) > 0) {    //保证diff[i]≠0，即分母不会为0

      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t) min_t = t1;

      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t) min_t = t2;
    }
  }
 
  // 利用相似三角形计算出的缩放比例系数
  return camera_pt + (min_t - 1e-3) * diff;
}

/* 这一函数首先将局部范围外一圈的点的occupancy_buffer对应值置为：mp_.clamp_min_log_ - mp_.unknown_flag_。
然后将局部地图范围内的地图上一轮的occupancy_buffer_inflate值全部置为0；
紧接着，对局部地图的occupancy_buffer中所有点的值进行一一判断，判断是否超过为障碍物的最低概率mp_.min_occupancy_log_，
若是，就对该点进行膨胀，并将所有膨胀点的occupancy_buffer_inflate值全部置为1。 */
void SDFMap::clearAndInflateLocalMap() {
  /*clear outside local*/
  const int vec_margin = 5;  // 表示在外围边界上设置的边缘距离
  // Eigen::Vector3i min_vec_margin = min_vec - Eigen::Vector3i(vec_margin,
  // vec_margin, vec_margin); Eigen::Vector3i max_vec_margin = max_vec +
  // Eigen::Vector3i(vec_margin, vec_margin, vec_margin);

  // local_bound_min_表示更新栅格的范围,local_map_margin_ = 50 表示原始局部地图的扩展距离
  Eigen::Vector3i min_cut = md_.local_bound_min_ -
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  Eigen::Vector3i max_cut = md_.local_bound_max_ +
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  boundIndex(min_cut);
  boundIndex(max_cut);

  Eigen::Vector3i min_cut_m = min_cut - Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  Eigen::Vector3i max_cut_m = max_cut + Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  boundIndex(min_cut_m);
  boundIndex(max_cut_m);

  // clear data outside the local range

  for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
    for (int y = min_cut_m(1); y <= max_cut_m(1); ++y) {

      for (int z = min_cut_m(2); z < min_cut(2); ++z) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int z = max_cut(2) + 1; z <= max_cut_m(2); ++z) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
    for (int x = min_cut_m(0); x <= max_cut_m(0); ++x) {

      for (int y = min_cut_m(1); y < min_cut(1); ++y) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int y = max_cut(1) + 1; y <= max_cut_m(1); ++y) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
    for (int z = min_cut_m(2); z <= max_cut_m(2); ++z) {

      for (int x = min_cut_m(0); x < min_cut(0); ++x) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int x = max_cut(0) + 1; x <= max_cut_m(0); ++x) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  // inflate occupied voxels to compensate robot size

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  // int inf_step_z = 1;
  // 各向同性膨胀，一个体素膨胀后仍然是立方体。2 * inf_step + 1是计算膨胀后的大立方体的边长上的体素数量，取3次方后就是大立方体全部的体素数量
  // pow(2 * inf_step + 1, 3)是一个立方体膨胀后的体素数量
  vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));   //数组中存储一个体素膨胀后，大立方中的每个栅格的索引
  // inf_pts.resize(4 * inf_step + 3);
  Eigen::Vector3i inf_pt;

  // clear outdated data
  for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
    for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z) {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
      }

  // inflate obstacles
  for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
    for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z) {

        if (md_.occupancy_buffer_[toAddress(x, y, z)] > mp_.min_occupancy_log_) {
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);  // 膨胀后的栅格点的坐标存在inf_pts中
          
          // 判断膨胀后的点是否在地图内，若在将该膨胀点的occupancy_buffer_inflate值置为1
          for (int k = 0; k < (int)inf_pts.size(); ++k) {
            inf_pt = inf_pts[k];
            int idx_inf = toAddress(inf_pt);
            if (idx_inf < 0 ||
                idx_inf >= mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2)) {
              continue;
            }
            md_.occupancy_buffer_inflate_[idx_inf] = 1;
          }
        }
      }

  // add virtual ceiling to limit flight height
  // mp_.virtual_ceil_height_=2.5, mp_.map_origin_(2) = mp_.ground_height_=-1.0
  if (mp_.virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_);  //得到虚拟天花板对应的高度索引
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y) {
        md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
      }
  }
}

void SDFMap::visCallback(const ros::TimerEvent& /*event*/) {
  publishMap();
  publishMapInflate(false);// 
  publishUpdateRange(); //
  publishESDF();       //

  // publishUnknown();
  publishDepth();//
}

void SDFMap::updateOccupancyCallback(const ros::TimerEvent& /*event*/) {
  if (!md_.occ_need_update_) return;

  /* update occupancy */
  ros::Time t1, t2;
  t1 = ros::Time::now();

  projectDepthImage();
  raycastProcess();

  if (md_.local_updated_) clearAndInflateLocalMap();

  t2 = ros::Time::now();

  md_.fuse_time_ += (t2 - t1).toSec();
  md_.max_fuse_time_ = max(md_.max_fuse_time_, (t2 - t1).toSec());

  if (mp_.show_occ_time_)
    ROS_WARN("Fusion: cur t = %lf, avg t = %lf, max t = %lf", (t2 - t1).toSec(),
             md_.fuse_time_ / md_.update_num_, md_.max_fuse_time_);

  md_.occ_need_update_ = false;
  if (md_.local_updated_) md_.esdf_need_update_ = true;
  md_.local_updated_ = false;
}

void SDFMap::updateESDFCallback(const ros::TimerEvent& /*event*/) {
  if (!md_.esdf_need_update_) return;

  /* esdf */
  ros::Time t1, t2;
  t1 = ros::Time::now();

  updateESDF3d();

  t2 = ros::Time::now();

  md_.esdf_time_ += (t2 - t1).toSec();
  md_.max_esdf_time_ = max(md_.max_esdf_time_, (t2 - t1).toSec());

  if (mp_.show_esdf_time_)
    ROS_WARN("ESDF: cur t = %lf, avg t = %lf, max t = %lf", (t2 - t1).toSec(),
             md_.esdf_time_ / md_.update_num_, md_.max_esdf_time_);

  md_.esdf_need_update_ = false;
}

void SDFMap::depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                               const geometry_msgs::PoseStampedConstPtr& pose) {
  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(md_.depth_image_);

  // std::cout << "depth: " << md_.depth_image_.cols << ", " << md_.depth_image_.rows << std::endl;

  /* get pose */
  md_.camera_pos_(0) = pose->pose.position.x;
  md_.camera_pos_(1) = pose->pose.position.y;
  md_.camera_pos_(2) = pose->pose.position.z;
  md_.camera_q_ = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
                                     pose->pose.orientation.y, pose->pose.orientation.z);
  if (isInMap(md_.camera_pos_)) {
    md_.has_odom_ = true;
    md_.update_num_ += 1;
    md_.occ_need_update_ = true;
  } else {
    md_.occ_need_update_ = false;
  }
}

void SDFMap::odomCallback(const nav_msgs::OdometryConstPtr& odom) {
  if (md_.has_first_depth_) return;

  md_.camera_pos_(0) = odom->pose.pose.position.x;
  md_.camera_pos_(1) = odom->pose.pose.position.y;
  md_.camera_pos_(2) = odom->pose.pose.position.z;

  md_.has_odom_ = true;
}

void SDFMap::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& img) {

  pcl::PointCloud<pcl::PointXYZ> latest_cloud;
  pcl::fromROSMsg(*img, latest_cloud);   // 将ROS消息转换为pcl点云库中消息格式

  md_.has_cloud_ = true;

  if (!md_.has_odom_) {
    // std::cout << "no odom!" << std::endl;
    return;
  }

  if (latest_cloud.points.size() == 0) return;

  if (isnan(md_.camera_pos_(0)) || isnan(md_.camera_pos_(1)) || isnan(md_.camera_pos_(2))) return;

  this->resetBuffer(md_.camera_pos_ - mp_.local_update_range_,
                    md_.camera_pos_ + mp_.local_update_range_);

  pcl::PointXYZ pt;
  Eigen::Vector3d p3d, p3d_inf;
  
  // 障碍物膨胀步长，即以膨胀点坐标为原点，水平面上前后左右各膨胀了几个体素
  // 竖直方向的膨胀步长，单独设置，可与水平方向的膨胀大小不同
  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_); 
  int inf_step_z = 1;   // inf_step_z = 1，表示上下各膨胀了一个体素

  double max_x, max_y, max_z, min_x, min_y, min_z;

  min_x = mp_.map_max_boundary_(0);  // 这样赋值，便于后续确定地图更新区域的最大最小边界
  min_y = mp_.map_max_boundary_(1);
  min_z = mp_.map_max_boundary_(2);

  max_x = mp_.map_min_boundary_(0);
  max_y = mp_.map_min_boundary_(1);
  max_z = mp_.map_min_boundary_(2);

  for (size_t i = 0; i < latest_cloud.points.size(); ++i) {
    pt = latest_cloud.points[i];
    p3d(0) = pt.x, p3d(1) = pt.y, p3d(2) = pt.z;

    /* point inside update range */
    Eigen::Vector3d devi = p3d - md_.camera_pos_;  // deviation 偏差、差值
    Eigen::Vector3i inf_pt;

    // mp_.local_update_range_x=5.5,y=5.5,z=4.5
    // 满足条件说明点p3d在地图的局部更新范围内
    if (fabs(devi(0)) < mp_.local_update_range_(0) && fabs(devi(1)) < mp_.local_update_range_(1) &&
        fabs(devi(2)) < mp_.local_update_range_(2)) {

      /* inflate the point */ 
      // 以膨胀点坐标为原点，三维空间中前后左右上下各膨胀了几个体素（体素的尺寸即地图分辨率）
      for (int x = -inf_step; x <= inf_step; ++x)
        for (int y = -inf_step; y <= inf_step; ++y)
          for (int z = -inf_step_z; z <= inf_step_z; ++z) {

            p3d_inf(0) = pt.x + x * mp_.resolution_;
            p3d_inf(1) = pt.y + y * mp_.resolution_;
            p3d_inf(2) = pt.z + z * mp_.resolution_;
           
            // 记录膨胀后点云区域的最大和最小边界值，便于后续确定地图更新区域的有效范围
            max_x = max(max_x, p3d_inf(0));
            max_y = max(max_y, p3d_inf(1));
            max_z = max(max_z, p3d_inf(2));

            min_x = min(min_x, p3d_inf(0));
            min_y = min(min_y, p3d_inf(1));
            min_z = min(min_z, p3d_inf(2));

            posToIndex(p3d_inf, inf_pt);

            if (!isInMap(inf_pt)) continue;

            int idx_inf = toAddress(inf_pt);

            md_.occupancy_buffer_inflate_[idx_inf] = 1;  //将膨胀后该点所对应的属性设置为1，表示该点已被膨胀
          }
    }
  }

  min_x = min(min_x, md_.camera_pos_(0));
  min_y = min(min_y, md_.camera_pos_(1));
  min_z = min(min_z, md_.camera_pos_(2));

  max_x = max(max_x, md_.camera_pos_(0));
  max_y = max(max_y, md_.camera_pos_(1));
  max_z = max(max_z, md_.camera_pos_(2));

  max_z = max(max_z, mp_.ground_height_);

  // 确定地图有效范围
  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_.local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_.local_bound_min_);

  boundIndex(md_.local_bound_min_);
  boundIndex(md_.local_bound_max_);

  md_.esdf_need_update_ = true;
}

void SDFMap::publishMap() {
  // pcl::PointXYZ pt;
  // pcl::PointCloud<pcl::PointXYZ> cloud;

  // Eigen::Vector3i min_cut = md_.local_bound_min_ -
  //     Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  // Eigen::Vector3i max_cut = md_.local_bound_max_ +
  //     Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);

  // boundIndex(min_cut);
  // boundIndex(max_cut);

  // for (int x = min_cut(0); x <= max_cut(0); ++x)
  //   for (int y = min_cut(1); y <= max_cut(1); ++y)
  //     for (int z = min_cut(2); z <= max_cut(2); ++z) {

  //       if (md_.occupancy_buffer_[toAddress(x, y, z)] <= mp_.min_occupancy_log_) continue;

  //       Eigen::Vector3d pos;
  //       indexToPos(Eigen::Vector3i(x, y, z), pos);
  //       if (pos(2) > mp_.visualization_truncate_height_) continue;

  //       pt.x = pos(0);
  //       pt.y = pos(1);
  //       pt.z = pos(2);
  //       cloud.points.push_back(pt);
  //     }

  // cloud.width = cloud.points.size();
  // cloud.height = 1;
  // cloud.is_dense = true;
  // cloud.header.frame_id = mp_.frame_id_;

  // sensor_msgs::PointCloud2 cloud_msg;
  // pcl::toROSMsg(cloud, cloud_msg);
  // map_pub_.publish(cloud_msg);

  // ROS_INFO("pub map");

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  int lmm = mp_.local_map_margin_ / 2;
  min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
  max_cut += Eigen::Vector3i(lmm, lmm, lmm);

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 0) continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_) continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_pub_.publish(cloud_msg);
}

void SDFMap::publishMapInflate(bool all_info) {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  if (all_info) {
    int lmm = mp_.local_map_margin_;
    min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
    max_cut += Eigen::Vector3i(lmm, lmm, lmm);
  }

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 0) continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_) continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_inf_pub_.publish(cloud_msg);

  // ROS_INFO("pub map");
}

void SDFMap::publishUnknown() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  boundIndex(max_cut);
  boundIndex(min_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {

        if (md_.occupancy_buffer_[toAddress(x, y, z)] < mp_.clamp_min_log_ - 1e-3) {
          Eigen::Vector3d pos;
          indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > mp_.visualization_truncate_height_) continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;

  // auto sz = max_cut - min_cut;
  // std::cout << "unknown ratio: " << cloud.width << "/" << sz(0) * sz(1) * sz(2) << "="
  //           << double(cloud.width) / (sz(0) * sz(1) * sz(2)) << std::endl;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

void SDFMap::publishDepth() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  for (int i = 0; i < md_.proj_points_cnt; ++i) {
    pt.x = md_.proj_points_[i][0];
    pt.y = md_.proj_points_[i][1];
    pt.z = md_.proj_points_[i][2];
    cloud.push_back(pt);
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  depth_pub_.publish(cloud_msg);
}

void SDFMap::publishUpdateRange() {
  Eigen::Vector3d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
  visualization_msgs::Marker mk;
  indexToPos(md_.local_bound_min_, esdf_min_pos);
  indexToPos(md_.local_bound_max_, esdf_max_pos);

  cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);   // cube几何中心的位置
  cube_scale = esdf_max_pos - esdf_min_pos;         // cube几何大小
  mk.header.frame_id = mp_.frame_id_;
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = 0;

  mk.pose.position.x = cube_pos(0);
  mk.pose.position.y = cube_pos(1);
  mk.pose.position.z = cube_pos(2);

  mk.scale.x = cube_scale(0);
  mk.scale.y = cube_scale(1);
  mk.scale.z = cube_scale(2);

  mk.color.a = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;

  mk.pose.orientation.w = 1.0;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;

  update_range_pub_.publish(mk);
}

void SDFMap::publishESDF() {
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = 0.0;
  const double max_dist = 3.0;

  Eigen::Vector3i min_cut = md_.local_bound_min_ -
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  Eigen::Vector3i max_cut = md_.local_bound_max_ +
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {

      Eigen::Vector3d pos;
      indexToPos(Eigen::Vector3i(x, y, 1), pos);
      pos(2) = mp_.esdf_slice_height_;    //指定需要显示ESDF信息的高度层位置

      dist = getDistance(pos);
      dist = min(dist, max_dist);  //出于可视化效果考量，将dist的数值限制在[min_dist,max_dist]范围内
      dist = max(dist, min_dist);

      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = -0.2;    //rviz可视化时点云所在的实际高度，其显示的是mp_.esdf_slice_height_指定高度上的esdf信息
      pt.intensity = (dist - min_dist) / (max_dist - min_dist); //距离归一化
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  esdf_pub_.publish(cloud_msg);

  // ROS_INFO("pub esdf");
}

void SDFMap::getSliceESDF(const double height, const double res, const Eigen::Vector4d& range,
                          vector<Eigen::Vector3d>& slice, vector<Eigen::Vector3d>& grad, int sign) {
  double dist;
  Eigen::Vector3d gd;
  for (double x = range(0); x <= range(1); x += res)
    for (double y = range(2); y <= range(3); y += res) {

      dist = this->getDistWithGradTrilinear(Eigen::Vector3d(x, y, height), gd);
      slice.push_back(Eigen::Vector3d(x, y, dist));
      grad.push_back(gd);
    }
}

void SDFMap::checkDist() {
  for (int x = 0; x < mp_.map_voxel_num_(0); ++x)
    for (int y = 0; y < mp_.map_voxel_num_(1); ++y)
      for (int z = 0; z < mp_.map_voxel_num_(2); ++z) {
        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);

        Eigen::Vector3d grad;
        double dist = getDistWithGradTrilinear(pos, grad);

        if (fabs(dist) > 10.0) {
        }
      }
}

bool SDFMap::odomValid() { return md_.has_odom_; }

bool SDFMap::hasDepthObservation() { return md_.has_first_depth_; }

double SDFMap::getResolution() { return mp_.resolution_; }

Eigen::Vector3d SDFMap::getOrigin() { return mp_.map_origin_; }

int SDFMap::getVoxelNum() {
  return mp_.map_voxel_num_[0] * mp_.map_voxel_num_[1] * mp_.map_voxel_num_[2];
}

void SDFMap::getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size) {
  ori = mp_.map_origin_, size = mp_.map_size_;
}

void SDFMap::getSurroundPts(const Eigen::Vector3d& pos, Eigen::Vector3d pts[2][2][2],
                            Eigen::Vector3d& diff) {
  if (!isInMap(pos)) {
    // cout << "pos invalid for interpolation." << endl;
  }

  /* interpolation position */
  Eigen::Vector3d pos_m = pos - 0.5 * mp_.resolution_ * Eigen::Vector3d::Ones();
  Eigen::Vector3i idx;
  Eigen::Vector3d idx_pos;

  posToIndex(pos_m, idx);
  indexToPos(idx, idx_pos);
  diff = (pos - idx_pos) * mp_.resolution_inv_;

  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
        Eigen::Vector3d current_pos;
        indexToPos(current_idx, current_pos);
        pts[x][y][z] = current_pos;
      }
    }
  }
}

void SDFMap::depthOdomCallback(const sensor_msgs::ImageConstPtr& img,
                               const nav_msgs::OdometryConstPtr& odom) {
  /* get pose */
  md_.camera_pos_(0) = odom->pose.pose.position.x;
  md_.camera_pos_(1) = odom->pose.pose.position.y;
  md_.camera_pos_(2) = odom->pose.pose.position.z;
  md_.camera_q_ = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                                     odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);

  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(md_.depth_image_);

  md_.occ_need_update_ = true;
}

void SDFMap::depthCallback(const sensor_msgs::ImageConstPtr& img) {
  std::cout << "depth: " << img->header.stamp << std::endl;
}

void SDFMap::poseCallback(const geometry_msgs::PoseStampedConstPtr& pose) {
  std::cout << "pose: " << pose->header.stamp << std::endl;

  md_.camera_pos_(0) = pose->pose.position.x;
  md_.camera_pos_(1) = pose->pose.position.y;
  md_.camera_pos_(2) = pose->pose.position.z;
}

// SDFMap
