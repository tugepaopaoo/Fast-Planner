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

#include <path_searching/kinodynamic_astar.h>
#include <sstream>
#include <plan_env/sdf_map.h>

using namespace std;
using namespace Eigen;

namespace fast_planner
{
KinodynamicAstar::~KinodynamicAstar()
{
  for (int i = 0; i < allocate_num_; i++)
  {
    delete path_node_pool_[i];
  }
}

int KinodynamicAstar::search(Eigen::Vector3d start_pt, Eigen::Vector3d start_v, Eigen::Vector3d start_a,
                             Eigen::Vector3d end_pt, Eigen::Vector3d end_v, bool init, bool dynamic, double time_start)
{
  /* ---------- (1)将起始点和终点的三维位置转化为栅格地图的index,并计算第一个扩展点的Heuristic cost ---------- */
  /* ---------- get traj of one shot ---------- */
  start_vel_ = start_v;
  start_acc_ = start_a;
  
  // 设置起点
  PathNodePtr cur_node = path_node_pool_[0];  //在初始化中已经为搜索过程所用的节点分配好了内存
  cur_node->parent = NULL;
  cur_node->state.head(3) = start_pt;
  cur_node->state.tail(3) = start_v;
  cur_node->index = posToIndex(start_pt);
  cur_node->g_score = 0.0;

  // 设置终点
  Eigen::VectorXd end_state(6);
  Eigen::Vector3i end_index;
  double time_to_goal;

  end_state.head(3) = end_pt;
  end_state.tail(3) = end_v;
  end_index = posToIndex(end_pt);
  cur_node->f_score = lambda_heu_ * estimateHeuristic(cur_node->state, end_state, time_to_goal);
  cur_node->node_state = IN_OPEN_SET;
  open_set_.push(cur_node);  //把起点放入open_list和expand_nodes_
  use_node_num_ += 1;

  if (dynamic)
  {
    time_origin_ = time_start;
    cur_node->time = time_start;
    cur_node->time_idx = timeToIndex(time_start);
    expanded_nodes_.insert(cur_node->index, cur_node->time_idx, cur_node);
    // cout << "time start: " << time_start << endl;
  }
  else
    expanded_nodes_.insert(cur_node->index, cur_node);

  PathNodePtr neighbor = NULL;
  PathNodePtr terminate_node = NULL;
  bool init_search = init;
  const int tolerance = ceil(1 / resolution_); //地图resolution_=0.1，tolerance用来判断是否near_end,阈值为1m（10 index）

/* --------------- (2)搜索扩张节点，并迭代循环 --------------- */
/* --------------- search loop --------------- */
  while (!open_set_.empty())
  {
    /* --------------- get lowest f_score node --------------- */
    cur_node = open_set_.top();

    // Terminate?
    // reach_horizon判断是否到达局部设定的局部规划范围，near_end判断是否到达终点附近
    bool reach_horizon = (cur_node->state.head(3) - start_pt).norm() >= horizon_; //horizon_由launch文件中设定

    /* ---------------determine termination--------------- */
    bool near_end = abs(cur_node->index(0) - end_index(0)) <= tolerance &&
                    abs(cur_node->index(1) - end_index(1)) <= tolerance &&
                    abs(cur_node->index(2) - end_index(2)) <= tolerance;
    
    if (reach_horizon || near_end)
    {
      terminate_node = cur_node;
      retrievePath(terminate_node);   //路径存在path_nodes_中
      if (near_end)
      {
        // Check whether shot traj exist
        // one shot trajectory
        estimateHeuristic(cur_node->state, end_state, time_to_goal);
        computeShotTraj(cur_node->state, end_state, time_to_goal);
        if (init_search)
          ROS_ERROR("Shot in first search loop!");
      }
    }
    if (reach_horizon)
    {
      if (is_shot_succ_)
      {
        std::cout << "reach end" << std::endl;
        return REACH_END;
      }
      else
      {
        std::cout << "reach horizon" << std::endl;
        return REACH_HORIZON;
      }
    }

    if (near_end)
    {
      if (is_shot_succ_)                  //若在终点附近找到了一段轨迹
      {
        std::cout << "reach end" << std::endl;
        return REACH_END;
      }
      else if (cur_node->parent != NULL)  //若在终点附近未找到轨迹，且当前节点有父节点
      {
        std::cout << "near end" << std::endl;
        return NEAR_END;
      }
      else
      {
        std::cout << "no path" << std::endl;
        return NO_PATH;
      }
    }

    /* ---------------若当前节点没有抵达终点，则需要进行节点扩张--------------- */
    /* ---------------pop node and add to close set--------------- */
    open_set_.pop();                        //先top返回队首的值，再pop删除队首的值
    cur_node->node_state = IN_CLOSE_SET;    //将当前访问过的节点加入close_set
    iter_num_ += 1;

    /* ---------------init state propagation--------------- */
    double res = 1 / 2.0;               //输入加速度的分辨率
    double time_res = 1 / 1.0;          //第二次及以后前向积分的时间分辨率
    double time_res_init = 1 / 20.0;    //第一次前向积分的时间分辨率

    Eigen::Matrix<double, 6, 1> cur_state = cur_node->state;
    Eigen::Matrix<double, 6, 1> pro_state;
    vector<PathNodePtr> tmp_expand_nodes;
    Eigen::Vector3d um;
    double pro_t;

    vector<Eigen::Vector3d> inputs;
    vector<double> durations;

    if (init_search)  //先按照之前的输入，规划一段路径，若不满足条件，则划分加速度前向积分获得新的路径
    {
      // FSM->GEN_NEW_TRAJ: start_acc_ = 0, FSM->REPLAN_TRAJ: start_acc_为当前机器人的加速度
      inputs.push_back(start_acc_);
      // for(t = 0.1, t <= 0.8, t += 0.1)
      for (double tau = time_res_init * init_max_tau_; tau <= init_max_tau_ + 1e-3;
           tau += time_res_init * init_max_tau_)
        durations.push_back(tau);    // tau: 每段轨迹前向积分的时间分辨率
      init_search = false;
    }
    else  //按之前的加速度进行积分得到的轨迹不满足条件，细分加速度后重新规划
    {
      // max_acc_ = 1 m/s^2, res = 0.5, input: -1, -0.5, 0, 0.5, 1
      for (double ax = -max_acc_; ax <= max_acc_ + 1e-3; ax += max_acc_ * res)
        for (double ay = -max_acc_; ay <= max_acc_ + 1e-3; ay += max_acc_ * res)
          for (double az = -max_acc_; az <= max_acc_ + 1e-3; az += max_acc_ * res)
          {
            um << ax, ay, az;       //每一小段轨迹的输入
            inputs.push_back(um);
          }

      // for(t = 0.1, t <= 0.8, t += 0.1)   每次前向积分的时间T = 0.1s    
      for (double tau = time_res * max_tau_; tau <= max_tau_; tau += time_res * max_tau_)
        durations.push_back(tau);
    }
     
    /* ---------------state propagation loop--------------- */
    // cout << "cur state:" << cur_state.head(3).transpose() << endl;
    for (int i = 0; i < inputs.size(); ++i)
      for (int j = 0; j < durations.size(); ++j)       //每个delta_t后的状态作为一个Node
      {
        um = inputs[i];
        double tau = durations[j];
        stateTransit(cur_state, pro_state, um, tau);   //每个delta_t的状态转换
        pro_t = cur_node->time + tau;                  //used for dynamic

        Eigen::Vector3d pro_pos = pro_state.head(3);

        // Check if in close set
        Eigen::Vector3i pro_id = posToIndex(pro_pos);
        int pro_t_id = timeToIndex(pro_t);

        // 在hash table(expanded_nodes_)中查找索引为pro_id的节点
        PathNodePtr pro_node = dynamic ? expanded_nodes_.find(pro_id, pro_t_id) : expanded_nodes_.find(pro_id);
        if (pro_node != NULL && pro_node->node_state == IN_CLOSE_SET)   //已经扩展,且访问过pro_node
        {
          if (init_search)
            std::cout << "close" << std::endl;
          continue;
        }

        // Check maximal velocity
        Eigen::Vector3d pro_v = pro_state.tail(3);
        if (fabs(pro_v(0)) > max_vel_ || fabs(pro_v(1)) > max_vel_ || fabs(pro_v(2)) > max_vel_)
        {
          if (init_search)
            std::cout << "vel" << std::endl;
          continue;
        }

        // Check not in the same voxel
        Eigen::Vector3i diff = pro_id - cur_node->index;
        int diff_time = pro_t_id - cur_node->time_idx;
        if (diff.norm() == 0 && ((!dynamic) || diff_time == 0))
        {
          if (init_search)
            std::cout << "same" << std::endl;
          continue;
        }

        // Check safety
        Eigen::Vector3d pos;
        Eigen::Matrix<double, 6, 1> xt;
        bool is_occ = false;
        for (int k = 1; k <= check_num_; ++k)
        {
          double dt = tau * double(k) / double(check_num_);   //check_num_ = 5, dt = 0.16s
          stateTransit(cur_state, xt, um, dt);
          pos = xt.head(3);
          if (edt_environment_->sdf_map_->getInflateOccupancy(pos) == 1 )
          {
            is_occ = true;
            break;
          }
        }
        if (is_occ)
        {
          if (init_search)
            std::cout << "safe" << std::endl;
          continue;
        }

        /* ---------------compute cost--------------- */
        double time_to_goal, tmp_g_score, tmp_f_score;
        // 代价:(um.squaredNorm()+w_time_)*tau,分别表示能量损耗和运行时间
        // 计算当前节点的真实代价g值
        tmp_g_score = (um.squaredNorm() + w_time_) * tau + cur_node->g_score;
        // 计算当前节点的启发式代价f值
        tmp_f_score = tmp_g_score + lambda_heu_ * estimateHeuristic(pro_state, end_state, time_to_goal);

        // Compare nodes expanded from the same parent
        bool prune = false;    //剪枝标志位
        for (int j = 0; j < tmp_expand_nodes.size(); ++j)  //这里tmp_expand_nodes是存储不同acc,tau的前向积分扩展的节点
        {
          PathNodePtr expand_node = tmp_expand_nodes[j];
          //首先，判断当前节点是否与tmp_expand_nodes中某些节点在同一voxel中,若是则需要选择cost最低的进行剪枝
          if ((pro_id - expand_node->index).norm() == 0 && ((!dynamic) || pro_t_id == expand_node->time_idx))
          {
            prune = true;
            if (tmp_f_score < expand_node->f_score)
            {
              expand_node->f_score = tmp_f_score;
              expand_node->g_score = tmp_g_score;
              expand_node->state = pro_state;
              expand_node->input = um;
              expand_node->duration = tau;
              if (dynamic)
                expand_node->time = cur_node->time + tau;
            }
            break;
          }
        }

        /* ---------------new neighbor in this loop--------------- */
        // This node end up in a voxel different from others
        if (!prune)
        {
          if (pro_node == NULL)   //pro_node为expanded_nodes_.find()的结果,pro_node == NULL说明是一个新节点
          {                       //expanded_nodes_对于已经扩展过的节点,不需重复分配内存,直接在哈希表中查找即可
            pro_node = path_node_pool_[use_node_num_];
            pro_node->index = pro_id;
            pro_node->state = pro_state;
            pro_node->f_score = tmp_f_score;
            pro_node->g_score = tmp_g_score;
            pro_node->input = um;
            pro_node->duration = tau;
            pro_node->parent = cur_node;
            pro_node->node_state = IN_OPEN_SET;
            if (dynamic)
            {
              pro_node->time = cur_node->time + tau;
              pro_node->time_idx = timeToIndex(pro_node->time);
            }
            open_set_.push(pro_node);

            if (dynamic)
              expanded_nodes_.insert(pro_id, pro_node->time, pro_node);
            else
              expanded_nodes_.insert(pro_id, pro_node);  //expanded_nodes_存储所有扩展过的节点,包括访问过的和待访问的

            // 在hash table中的节点只能插入和查找,无法遍历,所以这里定义一个vector来存储所有扩展过的节点,方便剪枝的时候遍历所有节点
            tmp_expand_nodes.push_back(pro_node);

            use_node_num_ += 1;
            if (use_node_num_ == allocate_num_)
            {
              cout << "run out of memory." << endl;
              return NO_PATH;
            }
          }
          else if (pro_node->node_state == IN_OPEN_SET)   //当前临时节点已经被扩展过，但未访问（即在open集中)
          {
            if (tmp_g_score < pro_node->g_score)          //若当前路径cost更小，则对节点状态进行更新
            {
              // pro_node->index = pro_id;
              pro_node->state = pro_state;
              pro_node->f_score = tmp_f_score;
              pro_node->g_score = tmp_g_score;
              pro_node->input = um;
              pro_node->duration = tau;
              pro_node->parent = cur_node;
              if (dynamic)
                pro_node->time = cur_node->time + tau;
            }
          }
          else
          {
            cout << "error type in searching: " << pro_node->node_state << endl;
          }
        }
      }
    // init_search = false;
  }

  /* --------------- open_list已经遍历完成，没有搜索到路径 --------------- */
  cout << "open set empty, no path!" << endl;
  cout << "use node num: " << use_node_num_ << endl;
  cout << "iter num: " << iter_num_ << endl;
  return NO_PATH;
}

void KinodynamicAstar::setParam(ros::NodeHandle& nh)
{
  nh.param("search/max_tau", max_tau_, -1.0);                   //最大时间(s)
  nh.param("search/init_max_tau", init_max_tau_, -1.0);         //初始化时间(s)
  nh.param("search/max_vel", max_vel_, -1.0);                   //最大速度(m/s)
  nh.param("search/max_acc", max_acc_, -1.0);                   //最大加速度(m/s^2)
  nh.param("search/w_time", w_time_, -1.0);                     //花费时间的权重
  nh.param("search/horizon", horizon_, -1.0);                   //限制全局规划的距离，保证实时性
  nh.param("search/resolution_astar", resolution_, -1.0);       //空间分辨率
  nh.param("search/time_resolution", time_resolution_, -1.0);   //时间分辨率
  nh.param("search/lambda_heu", lambda_heu_, -1.0);             //启发函数的权重
  nh.param("search/allocate_num", allocate_num_, -1);           //最大节点数目
  nh.param("search/check_num", check_num_, -1);                 //对中间状态安全检查,检查数量
  nh.param("search/optimistic", optimistic_, true);
  tie_breaker_ = 1.0 + 1.0 / 10000;                             //用于轻微打破A*搜索时路径的对称性，加速搜索

  double vel_margin;
  nh.param("search/vel_margin", vel_margin, 0.0);
  max_vel_ += vel_margin;
}

void KinodynamicAstar::retrievePath(PathNodePtr end_node)       //获取路径
{
  PathNodePtr cur_node = end_node;
  path_nodes_.push_back(cur_node);

  while (cur_node->parent != NULL)
  {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node);
  }

  reverse(path_nodes_.begin(), path_nodes_.end());
}

// 估计激发函数成本
// 利用庞特里亚金原理解决两点边界值问题，得到的最优解后用最优解的控制代价作为启发函数。
double KinodynamicAstar::estimateHeuristic(Eigen::VectorXd x1, Eigen::VectorXd x2, double& optimal_time)
{
  const Vector3d dp = x2.head(3) - x1.head(3);    //两点间的距离
  const Vector3d v0 = x1.segment(3, 3);           //起点速度
  const Vector3d v1 = x2.segment(3, 3);           //终点速速

  // (1)首先通过设置启发函数对时间求导等于0，得到启发函数关于T的一元四次方程
  double c1 = -36 * dp.dot(dp);
  double c2 = 24 * (v0 + v1).dot(dp);
  double c3 = -4 * (v0.dot(v0) + v0.dot(v1) + v1.dot(v1));
  double c4 = 0;
  double c5 = w_time_;

  // (2)求解一元四次方程，得到一系列实根，并比较其对应的cost大小，得到最优时间
  std::vector<double> ts = quartic(c5, c4, c3, c2, c1);

  double v_max = max_vel_ * 0.5;
  double t_bar = (x1.head(3) - x2.head(3)).lpNorm<Infinity>() / v_max;
  ts.push_back(t_bar);

  double cost = 100000000;
  double t_d = t_bar;

  // 比较实根对应的cost大小，得到最优时间
  for (auto t : ts)
  {
    if (t < t_bar)
      continue;
    double c = -c1 / (3 * t * t * t) - c2 / (2 * t * t) - c3 / t + w_time_ * t;  //计算最优时间对应的控制代价
    if (c < cost)
    {
      cost = c;
      t_d = t;
    }
  }

  optimal_time = t_d;

  return 1.0 * (1 + tie_breaker_) * cost;
}

// 利用庞特里亚金原理求解一个两点边界值问题(OBVP问题)
// 最优控制时间已经在estimateHeuristic函数中计算得到，这里只需引入改时间进行多项式计算.
// 目的：验证轨迹是否安全，即不发生碰撞，速度、加速度不超过限制。
bool KinodynamicAstar::computeShotTraj(Eigen::VectorXd state1, Eigen::VectorXd state2, double time_to_goal)
{
  /* ---------- get coefficient ---------- */
  const Vector3d p0 = state1.head(3);
  const Vector3d dp = state2.head(3) - p0;
  const Vector3d v0 = state1.segment(3, 3);
  const Vector3d v1 = state2.segment(3, 3);
  const Vector3d dv = v1 - v0;
  double t_d = time_to_goal;
  MatrixXd coef(3, 4);
  end_vel_ = v1;

  Vector3d a = 1.0 / 6.0 * (-12.0 / (t_d * t_d * t_d) * (dp - v0 * t_d) + 6 / (t_d * t_d) * dv);  //a = 1/6 * alpha
  Vector3d b = 0.5 * (6.0 / (t_d * t_d) * (dp - v0 * t_d) - 2 / t_d * dv);                        //b = 1/2 * beta
  Vector3d c = v0;                                                                                //c = v0
  Vector3d d = p0;                                                                                //d = p0

  // 1/6 * alpha * t^3 + 1/2 * beta * t^2 + v0
  // a*t^3 + b*t^2 + v0*t + p0
  // coef从第一列开始分别为：t的0次项系数，1次项系数，2次项系数，3次项系数
  coef.col(3) = a, coef.col(2) = b, coef.col(1) = c, coef.col(0) = d;

  Vector3d coord, vel, acc;
  VectorXd poly1d, t, polyv, polya;
  Vector3i index;

  Eigen::MatrixXd Tm(4, 4);
  Tm << 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0;  //微分矩阵

  /* ---------- forward checking of trajectory ---------- */
  double t_delta = t_d / 10;
  for (double time = t_delta; time <= t_d; time += t_delta)
  {
    t = VectorXd::Zero(4);
    for (int j = 0; j < 4; j++)
      t(j) = pow(time, j);        //从0号元素依次是:t^0, t^1, t^2, t^3

    for (int dim = 0; dim < 3; dim++)        //根据离散的时间计算time点的coord, vel, acc
    {
      poly1d = coef.row(dim);
      coord(dim) = poly1d.dot(t);            //pos的一维数值
      vel(dim) = (Tm * poly1d).dot(t);       //vel的一维数值
      acc(dim) = (Tm * Tm * poly1d).dot(t);  //acc的一维数值

      if (fabs(vel(dim)) > max_vel_ || fabs(acc(dim)) > max_acc_)
      {
        // cout << "vel:" << vel(dim) << ", acc:" << acc(dim) << endl;
        // return false;
      }
    }

    // 判断坐标是否超出地图尺寸限制 
    if (coord(0) < origin_(0) || coord(0) >= map_size_3d_(0) || coord(1) < origin_(1) || coord(1) >= map_size_3d_(1) ||
        coord(2) < origin_(2) || coord(2) >= map_size_3d_(2))
    {
      return false;
    }

    // if (edt_environment_->evaluateCoarseEDT(coord, -1.0) <= margin_) {
    //   return false;
    // }
    if (edt_environment_->sdf_map_->getInflateOccupancy(coord) == 1)  //判断采样时刻该点是否在膨胀的障碍物中，不能保证轨迹一定安全
    {
      return false;
    }
  }
  coef_shot_ = coef;
  t_shot_ = t_d;
  is_shot_succ_ = true;
  return true;
}

// 求解一元三次方程，求解过程见wikipedia中“三次方程”里的求根系数法和三角函数法
// 判别式大于等于0时，利用求根公式；小于0，利用三角函数法
vector<double> KinodynamicAstar::cubic(double a, double b, double c, double d)
{
  vector<double> dts;

  double a2 = b / a;
  double a1 = c / a;
  double a0 = d / a;

  double Q = (3 * a1 - a2 * a2) / 9;
  double R = (9 * a1 * a2 - 27 * a0 - 2 * a2 * a2 * a2) / 54;
  double D = Q * Q * Q + R * R;
  if (D > 0)
  {
    double S = std::cbrt(R + sqrt(D));
    double T = std::cbrt(R - sqrt(D));
    dts.push_back(-a2 / 3 + (S + T));
    return dts;
  }
  else if (D == 0)
  {
    double S = std::cbrt(R);
    dts.push_back(-a2 / 3 + S + S);
    dts.push_back(-a2 / 3 - S);
    return dts;
  }
  else
  {
    double theta = acos(R / sqrt(-Q * Q * Q));
    dts.push_back(2 * sqrt(-Q) * cos(theta / 3) - a2 / 3);
    dts.push_back(2 * sqrt(-Q) * cos((theta + 2 * M_PI) / 3) - a2 / 3);
    dts.push_back(2 * sqrt(-Q) * cos((theta + 4 * M_PI) / 3) - a2 / 3);
    return dts;
  }
}

// 一元四次方程求解，求解过程见wikipedia中“费拉里方法”
vector<double> KinodynamicAstar::quartic(double a, double b, double c, double d, double e)
{
  vector<double> dts;

  double a3 = b / a;
  double a2 = c / a;
  double a1 = d / a;
  double a0 = e / a;

  vector<double> ys = cubic(1, -a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1 * a1 - a3 * a3 * a0);
  double y1 = ys.front();
  double r = a3 * a3 / 4 - a2 + y1;
  if (r < 0)
    return dts;

  double R = sqrt(r);
  double D, E;
  if (R != 0)
  {
    D = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 + 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
    E = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 - 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
  }
  else
  {
    D = sqrt(0.75 * a3 * a3 - 2 * a2 + 2 * sqrt(y1 * y1 - 4 * a0));
    E = sqrt(0.75 * a3 * a3 - 2 * a2 - 2 * sqrt(y1 * y1 - 4 * a0));
  }

  if (!std::isnan(D))
  {
    dts.push_back(-a3 / 4 + R / 2 + D / 2);
    dts.push_back(-a3 / 4 + R / 2 - D / 2);
  }
  if (!std::isnan(E))
  {
    dts.push_back(-a3 / 4 - R / 2 + E / 2);
    dts.push_back(-a3 / 4 - R / 2 - E / 2);
  }

  return dts;
}

void KinodynamicAstar::init()
{
  /* ---------- map params ---------- */
  this->inv_resolution_ = 1.0 / resolution_;
  inv_time_resolution_ = 1.0 / time_resolution_;
  edt_environment_->sdf_map_->getRegion(origin_, map_size_3d_);

  cout << "origin_: " << origin_.transpose() << endl;
  cout << "map size: " << map_size_3d_.transpose() << endl;

  /* ---------- pre-allocated node ---------- */
  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++)
  {
    path_node_pool_[i] = new PathNode;
  }

  phi_ = Eigen::MatrixXd::Identity(6, 6);
  use_node_num_ = 0;
  iter_num_ = 0;
}

void KinodynamicAstar::setEnvironment(const EDTEnvironment::Ptr& env)
{
  this->edt_environment_ = env;
}

void KinodynamicAstar::reset()
{
  expanded_nodes_.clear();
  path_nodes_.clear();

  std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> empty_queue;
  open_set_.swap(empty_queue);

  for (int i = 0; i < use_node_num_; i++)
  {
    PathNodePtr node = path_node_pool_[i];
    node->parent = NULL;
    node->node_state = NOT_EXPAND;
  }

  use_node_num_ = 0;
  iter_num_ = 0;
  is_shot_succ_ = false;
  has_path_ = false;
}

// 输入预设时间分辨率，获取对应的一段动力学轨迹state_list，仅用于可视化
std::vector<Eigen::Vector3d> KinodynamicAstar::getKinoTraj(double delta_t)
{
  vector<Vector3d> state_list;    //记录轨迹在采样点的pos

  /* ---------- get traj of searching ---------- */
  PathNodePtr node = path_nodes_.back();
  Matrix<double, 6, 1> x0, xt;    //pos_x, pos_y, pos_z, vel_x, vel_y, vel_z

  while (node->parent != NULL)    //以当前节点为终点的一小段轨迹上的input
  {
    Vector3d ut = node->input;
    double duration = node->duration;
    x0 = node->parent->state;

    for (double t = duration; t >= -1e-5; t -= delta_t)
    {
      stateTransit(x0, xt, ut, t);
      state_list.push_back(xt.head(3));
    }
    node = node->parent;
  }
  reverse(state_list.begin(), state_list.end());
  /* ---------- get traj of one shot ---------- */
  if (is_shot_succ_)
  {
    Vector3d coord;
    VectorXd poly1d, time(4);

    for (double t = delta_t; t <= t_shot_; t += delta_t)
    {
      for (int j = 0; j < 4; j++)
        time(j) = pow(t, j);

      for (int dim = 0; dim < 3; dim++)
      {
        poly1d = coef_shot_.row(dim);
        coord(dim) = poly1d.dot(time);
      }
      state_list.push_back(coord);
    }
  }

  return state_list;
}

// KinodynamicAstar采样，离散获得一些轨迹点及起始点速度和加速度，得到节点point_set，即open_list
// 获得的轨迹点用于B样条的拟合，以及后续的轨迹优化
void KinodynamicAstar::getSamples(double& ts, vector<Eigen::Vector3d>& point_set,
                                  vector<Eigen::Vector3d>& start_end_derivatives)
{
  /* ---------- path duration ---------- */
  double T_sum = 0.0;
  if (is_shot_succ_)
    T_sum += t_shot_;
  PathNodePtr node = path_nodes_.back();
  while (node->parent != NULL)
  {
    T_sum += node->duration;  //node->duration记录的是以当前节点为终点的前面一小段轨迹的总时长
    node = node->parent;
  }
  // cout << "duration:" << T_sum << endl;

  // Calculate boundary vel and acc
  Eigen::Vector3d end_vel, end_acc;   //设置终点的速度和加速度
  double t;
  if (is_shot_succ_)    //若能采样到轨迹点，使用运动约束计算终点的速度、终点的加速度
  {
    t = t_shot_;
    end_vel = end_vel_;
    for (int dim = 0; dim < 3; ++dim)
    {
      Vector4d coe = coef_shot_.row(dim);
      end_acc(dim) = 2 * coe(2) + 6 * coe(3) * t_shot_;
    }
  }
  else                 //若不能采样到轨迹点，使用手动输入的参数设置终点的速度、终点的加速度
  {
    t = path_nodes_.back()->duration;
    end_vel = node->state.tail(3);
    end_acc = path_nodes_.back()->input;
  }

  /* ---------- init for sampling ---------- */
  // Get point samples
  int seg_num = floor(T_sum / ts);
  seg_num = max(8, seg_num);
  ts = T_sum / double(seg_num);
  bool sample_shot_traj = is_shot_succ_;
  node = path_nodes_.back();

  for (double ti = T_sum; ti > -1e-5; ti -= ts)
  {
    if (sample_shot_traj)         //若采样得到轨迹点，则存入point_set
    {
      // samples on shot traj
      Vector3d coord;
      Vector4d poly1d, time;

      for (int j = 0; j < 4; j++)
        time(j) = pow(t, j);

      for (int dim = 0; dim < 3; dim++)
      {
        poly1d = coef_shot_.row(dim);
        coord(dim) = poly1d.dot(time);  //计算time处的1d（1维度）坐标
      }

      point_set.push_back(coord);
      t -= ts;                          //t为shot traj的时间

      /* 分段的结尾 end of segment */
      if (t < -1e-5)
      {
        sample_shot_traj = false;
        if (node->parent != NULL)
          t += node->duration;
      }
    }
    else                        //若没有采样得到轨迹点，进行状态传递，继续采样
    {
      // samples on searched traj
      Eigen::Matrix<double, 6, 1> x0 = node->parent->state;
      Eigen::Matrix<double, 6, 1> xt;
      Vector3d ut = node->input;

      stateTransit(x0, xt, ut, t);

      point_set.push_back(xt.head(3));
      t -= ts;

      // cout << "t: " << t << ", t acc: " << T_accumulate << endl;
      if (t < -1e-5 && node->parent->parent != NULL)   //当前轨迹已经采样完，继续采样一段轨迹
      {
        node = node->parent;    //因为已经预先将轨迹用ts均分，所以最后不会出现起点不对齐的情况
        t += node->duration;    //递归地计算t是为了保持时间轴的统一
      }
    }
  }
  reverse(point_set.begin(), point_set.end());   //将时间轴由逆序，调整为正序

  // calculate start acc
  Eigen::Vector3d start_acc;
  if (path_nodes_.back()->parent == NULL)
  {
    // no searched traj, calculate by shot traj
    start_acc = 2 * coef_shot_.col(2);
  }
  else
  {
    // input of searched traj
    start_acc = node->input;
  }

  start_end_derivatives.push_back(start_vel_);
  start_end_derivatives.push_back(end_vel);
  start_end_derivatives.push_back(start_acc);
  start_end_derivatives.push_back(end_acc);
}

std::vector<PathNodePtr> KinodynamicAstar::getVisitedNodes()
{
  vector<PathNodePtr> visited;
  visited.assign(path_node_pool_.begin(), path_node_pool_.begin() + use_node_num_ - 1);
  return visited;
}

Eigen::Vector3i KinodynamicAstar::posToIndex(Eigen::Vector3d pt)     //将三维位置转换为栅格地图的index
{
  Vector3i idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();

  // idx << floor((pt(0) - origin_(0)) * inv_resolution_), floor((pt(1) -
  // origin_(1)) * inv_resolution_),
  //     floor((pt(2) - origin_(2)) * inv_resolution_);

  return idx;
}

int KinodynamicAstar::timeToIndex(double time)      //在设置的时间段内，当前时间的索引
{
  int idx = floor((time - time_origin_) * inv_time_resolution_);
}

// 状态传递，通过前向积分得到扩展节点的位置和速度
void KinodynamicAstar::stateTransit(Eigen::Matrix<double, 6, 1>& state0, Eigen::Matrix<double, 6, 1>& state1,
                                    Eigen::Vector3d um, double tau)
{
  for (int i = 0; i < 3; ++i)
    phi_(i, i + 3) = tau;

  Eigen::Matrix<double, 6, 1> integral;
  integral.head(3) = 0.5 * pow(tau, 2) * um;
  integral.tail(3) = tau * um;

  // phi_ = 1, 0, 0, t, 0, 0      state0 = x0       integral = 0.5*ax*t^2         
  //        0, 1, 0, 0, t, 0               y0                  0.5*ay*t^2  
  //        0, 0, 1, 0, 0, t               z0                  0.5*az*t^2 
  //        0, 0, 0, 1, 0, 0               vx0                     ax*t
  //        0, 0, 0, 0, 1, 0               vy0                     ay*t
  //        0, 0, 0, 0, 0, 1               vz0                     az*t

  state1 = phi_ * state0 + integral;
}

}  // namespace fast_planner
