
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include "tf/tf.h"
#include <visualization_msgs/Marker.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/ExecuteKnownTrajectory.h>

#include <iostream>
#include <chrono>
#include <random>

static const std::string HI_RES_WINDOW = "High Resolution Camera";
static const std::string LO_RES_WINDOW = "Low Resolution Camera";
static const std::string REGION_WINDOW = "Selected Region Of Interest";
static const std::string FEATURE_WINDOW = "Feature Region";

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <map>
#include <functional>
#include <stdlib.h>
#include "/opt/ros/indigo/include/gazebo_msgs/DeleteModel.h"
#include "/opt/ros/indigo/include/gazebo_msgs/SpawnModel.h"

class pixl
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  ros::Subscriber imageHRSub;
  ros::Subscriber imageLRSub;
  ros::Subscriber matlabSub;
  ros::Subscriber servoSub;
  image_transport::Publisher watsonPub;
  image_transport::Publisher pixlPub;
  image_transport::Publisher servoPub;
  ros::Publisher scalePub;
  ros::Publisher spawnPub;
  sensor_msgs::ImagePtr watsonMsg;
  sensor_msgs::ImagePtr pixlMsg;
  sensor_msgs::ImagePtr servoMsg;
  sensor_msgs::Image points;
  std_msgs::Float32MultiArray scale;

  cv::Point pointA;
  cv::Point pointB;
  cv::Point featurePoint;
  std::ostringstream note;

  bool reset_pixl;
  bool reset_watson;

  std::vector<moveit_msgs::CollisionObject> collision_objects;

  std::vector<geometry_msgs::Pose> waypoints;

  tf::TransformListener tf_listener;
  float watsonScale;
  float pixlScale;
  cv::Mat color;
  cv::Mat grey;
  cv::Mat watson_initial_image;
  cv::Mat pixl_initial_image;
  cv::Mat pixl_est_image;
  cv::Mat regionOfInterest;
  cv::Mat featureOfInterest;

  geometry_msgs::PoseStamped object_pose;
  geometry_msgs::PoseStamped arm_pose;
  geometry_msgs::PoseStamped pixl_pose;
  geometry_msgs::PoseStamped watson_pose;
  geometry_msgs::PoseStamped est_pose;
  geometry_msgs::PoseStamped servo_pose;

  moveit::planning_interface::MoveGroup::Plan plan;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  moveit::planning_interface::MoveGroup move_group;
  std::string menuSelect;
  float objectposx;
  float multiplier;
  float multiplier2;
  float wait_time;
  std::vector<float> servo_data;
  bool reg_complete;
  bool reg_success;
  float servo_scale;
  float servo_rotation;
  float servo_z;
  float servo_y;
  int servo_attempts;
  bool servo_success;
  std::string operator_success;
  std::ofstream result_file;
  time_t timer;
  double time_id;
  bool servoing;
  std_msgs::String spawnMsg;
  bool initialise;

public:
  pixl()
  : it_(nh_), move_group("manipulator")
  {
    cv::namedWindow(HI_RES_WINDOW,cv::WINDOW_NORMAL);
    watsonPub = it_.advertise("/pixl/image_watson", 1);
    pixlPub = it_.advertise("/pixl/image_pixl", 1);
    servoPub = it_.advertise("/pixl/image_servo", 1);
    scalePub = nh_.advertise<std_msgs::Float32MultiArray>("/pixl/scale", 1);
    spawnPub = nh_.advertise<std_msgs::String>("/pixl/spawn", 1);

    std::stringstream ss;
    ss << "blah";
    spawnMsg.data = ss.str();

    init_scale();
    imageHRSub = nh_.subscribe("/camera_hires/image", 10, &pixl::callbackHR, this);
    matlabSub = nh_.subscribe("/pixl/matlab_result", 10, &pixl::matlabCallback, this);
    servoSub = nh_.subscribe("/pixl/servo_result", 10, &pixl::servoCallback, this);
    watsonScale = 1.0;
    pixlScale = 1.0;
    servoing = false;
    note.str("");
    // object at about 1m, use gaussian noise
    objectposx = create_random(1.0, 0.05);
    ROS_INFO_STREAM("Object estimated at: " << objectposx << " meters.");
    // addBoxObject(,1.0);

  }

  ~pixl()
  {

    std::vector<std::string> object_names = planning_scene_interface.getKnownObjectNames();
    planning_scene_interface.removeCollisionObjects(object_names);
    cv::destroyWindow(HI_RES_WINDOW);
    cv::destroyWindow(LO_RES_WINDOW);
    cv::destroyWindow(REGION_WINDOW);
    cv::destroyWindow(FEATURE_WINDOW);

  }

  void start(){

    std::string xInput;
    std::string yInput;
    std::string zInput;

    wait_time = 1.0;
    cv::setMouseCallback(HI_RES_WINDOW, mouse_click, this);

    reset_pixl = false;
    ros::Time time;

    move_group.setPlannerId("RRTConnectkConfigDefault");
    move_group.setPlanningTime(5);
    move_group.setMaxVelocityScalingFactor(0.75);
    move_group.setPoseReferenceFrame("world");
    // move_group.setEndEffectorLink("camera_hires_link");
    // ROS_INFO_STREAM(move_group.getEndEffectorLink());
    // ROS_INFO_STREAM(move_group.getEndEffectorLink());
    // move_group.setEndEffectorLink("ee_link");
    ros::AsyncSpinner spinner(2);
    spinner.start();
    initialise = true;


    while(ros::ok()){

      if(initialise){

        new_pixl();

        try {
          if(!moveTo(pixl_pose,2, wait_time)){throw tf::TransformException("Pixl pose initial failed.");}
        } catch (tf::TransformException ex){
          ROS_INFO_STREAM(ex.what());
        }
        // move_group.rememberJointValues("pixl_initial");

        new_watson();


        try {
          if(!moveTo(watson_pose,2, wait_time)){throw tf::TransformException("Watson pose initial failed.");}
        } catch (tf::TransformException ex){
          ROS_INFO_STREAM(ex.what());
        }

        // move_group.rememberJointValues("watson_initial");

        reset_pixl = false;

        reset_watson = false;
        initialise = false;

        new_example();
      }

      else if(reset_pixl){

        reset_pixl = false;
        try {
          if(!moveTo(pixl_pose,2, wait_time)){throw tf::TransformException("Pixl pose reset failed.");}
        } catch (tf::TransformException ex){
          ROS_INFO_STREAM(ex.what());
        }

      } else if (reset_watson) {

        reset_watson = false;
        try {
          if(!moveTo(watson_pose,2, wait_time)){throw tf::TransformException("Watson pose reset failed.");}
        } catch (tf::TransformException ex){
          ROS_INFO_STREAM(ex.what());
        }

      } else {
      }

    }

  } // End Start

  std::string dtos(double x) {
    std::stringstream s;
    s << x;
    return s.str();
  }

  void new_example() {



    spawnPub.publish(spawnMsg);

  }

  void new_pixl() {
    pixl_pose.pose.position.x = create_random(0.7,0.05);
    pixl_pose.pose.position.y = create_random(0.0,0.05);
    pixl_pose.pose.position.z = create_random(0.6,0.2);
    pixl_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, M_PI/2, M_PI);

    ROS_INFO("\n\nNew PIXL initial position\n\tX: %0.3f\n\tY: %0.3f\n\tZ: %0.3f\n", pixl_pose.pose.position.x, pixl_pose.pose.position.y, pixl_pose.pose.position.z);
    reset_pixl = true;
  }


  // void new_example()
  // {
  //   ros::service::waitForService("gazebo/spawn_sdf_model");
  //   ros::ServiceClient spawnModelClient = nh_.serviceClient<gazebo_msgs::SpawnModel>("gazebo/spawn_sdf_model");
  //   gazebo_msgs::SpawnModel spawnModel;
  //   spawnModel.request.model_name = std::string("pixl");
  //   spawnModelClient.call(spawnModel);
  // }


  void remove_example()
  {
    ros::service::waitForService("gazebo/delete_model");
    ros::ServiceClient deleteModelClient = nh_.serviceClient<gazebo_msgs::DeleteModel>("gazebo/delete_model");
    gazebo_msgs::DeleteModel deleteModel;
    deleteModel.request.model_name = std::string("pixl");
    deleteModelClient.call(deleteModel);
  }

  void new_watson() {

    watson_pose.pose.position.x = create_random(0.7,0.05);
    watson_pose.pose.position.y = create_random(0.0,0.05);
    watson_pose.pose.position.z = create_random(0.6,0.2);
    watson_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, M_PI/2, M_PI);

    ROS_INFO("\n\nNew Watson initial position\n\tX: %0.3f\n\tY: %0.3f\n\tZ: %0.3f\n", watson_pose.pose.position.x, watson_pose.pose.position.y, watson_pose.pose.position.z);
    reset_watson = true;
  }

  float create_random(float mean, float std) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (mean,std);
    return distribution(generator);
  }

  void init_scale() {
    scale.data.clear();
    scale.data.push_back(1);
    scale.data.push_back(1);
  }

  void updateSendScale() {
    scale.data.clear();
    if (watsonScale < pixlScale) {
      scale.data.push_back(watsonScale / pixlScale);
      scale.data.push_back(1);

    } else {

      scale.data.push_back(1);
      scale.data.push_back(pixlScale / watsonScale);
    }

    scalePub.publish(scale);

  }

  bool moveToNamed(std::string namedGoal, int Attempts, float timeout){

    bool ret = false;
    bool success = false;

    while(Attempts > 0){

      move_group.setStartStateToCurrentState();
      move_group.setNamedTarget(namedGoal);

      success = move_group.plan(plan);

      if(success){

        try{

          move_group.move();
          Attempts = 0;
          ret = true;

        }

        catch(moveit::planning_interface::MoveItErrorCode ex) {

          std::cout << "Something went wrong. Failed to move to Goal" << std::endl;
          ret = false;

        }

      } else {

        std::cout << "Planning unsuccessful" << std::endl;
        Attempts--;
        sleep(timeout);

      }

    }

    return ret;

  }

  bool moveTo(geometry_msgs::PoseStamped pose, int Attempts, float timeout){

    bool ret = false;
    bool success = false;

    while(Attempts > 0){

      move_group.setStartStateToCurrentState();
      pose.header.frame_id = "world";
      pose.header.stamp = ros::Time::now();
      move_group.setPoseTarget(pose);

      success = move_group.plan(plan);

      if(success){

        try{

          move_group.move();
          Attempts = 0;
          ret = true;

        }
        catch(moveit::planning_interface::MoveItErrorCode ex){

          std::cout << "Something went wrong. Failed to move to Goal" << std::endl;
          ret = false;

        }

      } else {

        std::cout << "Planning unsuccessful" << std::endl;
        Attempts--;
        sleep(timeout);

        if (servoing) {
          note << "Planning unsuccessful. ";
        }

      }
    }

    return ret;

  }

  void calculateTrajectory(int numSteps){

    geometry_msgs::Pose current = move_group.getCurrentPose().pose;

    float dx = (arm_pose.pose.position.x - current.position.x) / numSteps;
    float dy = (arm_pose.pose.position.y - current.position.y) / numSteps;
    float dz = (arm_pose.pose.position.z - current.position.z) / numSteps;

    for (int i=0;i<numSteps;i++){

      current.position.x += dx;
      current.position.y += dy;
      current.position.z += dz;

      waypoints.push_back(current);

    };

    waypoints.begin();
  }

  bool moveToCartesianPath(geometry_msgs::PoseStamped targetPose, int Attempts, float timeout, bool async){

    bool response = true;
    ros::ServiceClient executeKnownTrajectoryServiceClient = nh_.serviceClient<moveit_msgs::ExecuteKnownTrajectory>("/execute_kinematic_path");

    move_group.setPoseReferenceFrame("world");
    move_group.setStartStateToCurrentState();
    move_group.setPoseTarget(targetPose);

    // set waypoints for which to compute path
    std::vector<geometry_msgs::Pose> waypoints;
    waypoints.push_back(move_group.getCurrentPose().pose);
    waypoints.push_back(targetPose.pose);
    moveit_msgs::ExecuteKnownTrajectory srv;

    // compute cartesian path
    double ret = move_group.computeCartesianPath(waypoints, 0.1, 10000, srv.request.trajectory, false);
    if(ret < 0){
      // no path could be computed
      ROS_INFO_STREAM("Unable to compute Cartesian path!");
      response = false;
    } else if (ret < 1){
      // path started to be computed, but did not finish
      ROS_INFO_STREAM("Cartesian path computation finished " << ret * 100 << "% only!");
      response = false;
    }

    // send trajectory to arm controller
    srv.request.wait_for_execution = true;
    executeKnownTrajectoryServiceClient.call(srv);
    return response;
  }

  void addBoxObject(std::string id, float dim1, float dim2, float dim3, float x, float y, float z){

    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = move_group.getPlanningFrame();
    collision_object.id = id;

    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);

    primitive.dimensions[0] = dim1;
    primitive.dimensions[1] = dim2;
    primitive.dimensions[2] = dim3;

    geometry_msgs::Pose object_pose;

    object_pose.position.x = x;
    object_pose.position.y = y;
    object_pose.position.z = z;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(object_pose);
    collision_object.operation = collision_object.ADD;

    collision_objects.push_back(collision_object);

    ROS_INFO("Add an object into the world");
    planning_scene_interface.addCollisionObjects(collision_objects);

  }

  void addPlaneObject(std::string id, float x_coeff){

    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = move_group.getPlanningFrame();
    collision_object.id = id;

    shape_msgs::Plane plane;

    plane.coef[0] = x_coeff;
    plane.coef[1] = 0.0;
    plane.coef[2] = 0.0;
    plane.coef[2] = 0.0;

    geometry_msgs::Pose object_pose;

    object_pose.position.x = 1.0;
    object_pose.position.y = 0;
    object_pose.position.z = 0;

    collision_object.planes.push_back(plane);
    collision_object.plane_poses.push_back(object_pose);
    collision_object.operation = collision_object.ADD;

    collision_objects.push_back(collision_object);

    ROS_INFO("Add an object into the world");
    planning_scene_interface.addCollisionObjects(collision_objects);

  }

  void callbackHR(const sensor_msgs::Image::ConstPtr imageColor)
  {
    readImageColor(imageColor, color);
    cv::imshow(HI_RES_WINDOW, color);
    cv::waitKey(3);
  }

  void callbackLR(const sensor_msgs::Image::ConstPtr imageGrey)
  {
    readImageGrey(imageGrey, grey);
    cv::imshow(LO_RES_WINDOW, grey);
    cv::waitKey(3);
  }

  void matlabCallback(const std_msgs::Float32MultiArray msg)
  {
    servoing = true;
    ROS_INFO_STREAM("SeqSLAM information received from MATLAB");
    std::vector<float> data = msg.data;
    Eigen::Map<Eigen::MatrixXf> mat(data.data(), msg.layout.dim[0].size, msg.layout.dim[1].size);
    if (mat(0,0) == 0) {
      ROS_INFO_STREAM("Initial registration failed.");
      servoing = false;
      note << "Initial registration failed. ";
      return;
    }

    // move to estimated extension
    arm_pose = move_group.getCurrentPose();
    arm_pose.pose.position.x = objectposx - (watsonScale / mat(0,0)) - 0.17;

    ROS_INFO_STREAM("Moving to estimated extension");
    try{

      if(!moveTo(arm_pose,2, wait_time)){throw tf::TransformException("Move to estimated extension failed");}

    }
    catch (tf::TransformException ex){
      reset_pixl = true;
      reset_watson = true;
      note << ex.what();
      ROS_INFO_STREAM(ex.what());
      servoing = false;
      return;

    }

    // move to estimated pose
    arm_pose = move_group.getCurrentPose();
    arm_pose.pose.position.y -= watsonScale / mat(0,0) * mat(0,3) / grey.cols;
    arm_pose.pose.position.z -= watsonScale / mat(0,0) * mat(0,2) / grey.rows;

    ROS_INFO_STREAM("Moving to estimated pose");
    try{

      if(!moveTo(arm_pose,2, wait_time)){throw tf::TransformException("Move to estimated position failed");}

    }
    catch (tf::TransformException ex){

      reset_pixl = true;
      reset_watson = true;
      note << ex.what();
      ROS_INFO_STREAM(ex.what());
      servoing = false;
      return;

    }

    if (!servoing) {
      ROS_INFO_STREAM("Not servoing");
      return;
    }
    est_pose = move_group.getCurrentPose();
    grey.copyTo(pixl_est_image);
    servo();
  }

  void servo() {

    ROS_INFO_STREAM("Begin visual servoing");
    init_scale();
    scalePub.publish(scale);
    servo_attempts = 10;
    servo_success = false;
    reg_success = false;
    std::vector<float> temp(4,0);
    Eigen::ArrayXXf set_point(1,4);
    Eigen::ArrayXXf err_sum(1,4);
    Eigen::ArrayXXf d_err(1,4);
    Eigen::ArrayXXf lastErr(1,4);
    Eigen::ArrayXXf current(1,4);
    Eigen::ArrayXXf error(1,4);
    Eigen::ArrayXXf output(1,4);
    Eigen::ArrayXXf kp(1,4);
    kp << 0.05, 0.5, 0.00005, 0.00005;
    Eigen::ArrayXXf ki(1,4);
    Eigen::ArrayXXf kd(1,4);

    while(!servo_success) {

      arm_pose = move_group.getCurrentPose();

      servoMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grey).toImageMsg();
      servoPub.publish(servoMsg);
      reg_complete = false;
      while(!reg_complete) {

      }

      if (reg_success) {

        // /*Compute all the working error variables*/
        current << 1-servo_scale,-servo_rotation * M_PI/180,servo_y,servo_z;
        error = set_point - current;
        err_sum += error;
        d_err = (error - lastErr);
        //
        // /*Compute PID Output*/
        output = kp * error + ki * err_sum + kd * d_err;
        tf::Quaternion q(arm_pose.pose.orientation.x, arm_pose.pose.orientation.y, arm_pose.pose.orientation.z, arm_pose.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        arm_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll+output(0,1), 0.0, 0.0);
        arm_pose.pose.position.x += output(0,0);
        arm_pose.pose.position.y += output(0,2);
        arm_pose.pose.position.z += output(0,3);
        ROS_INFO_STREAM("Moving to servo position");
        try{

          if(!moveTo(arm_pose,2, wait_time)){throw tf::TransformException("Move to servo position failed");}

        }
        catch (tf::TransformException ex){

          note << ex.what();
          ROS_INFO_STREAM(ex.what());
          servo_success = true;
          return;

        }
        lastErr = error;
      }

    }
    std::cout << "Successful attempt [Y/n]? ";
    std::cin >> operator_success;
    servo_pose = move_group.getCurrentPose();
    save_result();
    servoing = false;

    remove_example();
    new_example();
    new_pixl();
    new_watson();
    new_estimate();
    reset_pixl = true;
    reset_watson = true;

  }

  void servoCallback(const std_msgs::Float32MultiArray msg)
  {

    ROS_INFO_STREAM("SURF information received from MATLAB");
    servo_data = msg.data;
    Eigen::Map<Eigen::MatrixXf> mat(servo_data.data(), msg.layout.dim[0].size, msg.layout.dim[1].size);
    reg_success = true ? mat(0,0) : false;
    if (reg_success) {
      servo_scale = mat(0,1);
      servo_rotation = mat(0,2);
      servo_z = mat(0,3);
      servo_y = mat(0,4);
      servo_attempts = 10;
      if (abs(1-servo_scale) < 0.01 && abs(servo_rotation) < 0.01 && abs(servo_z) < 5 && abs(servo_y) < 5) {
        servo_success = true;
        ROS_INFO_STREAM("Servoing successful");
      }
    } else {
      servo_attempts -= 1;
      ROS_INFO_STREAM("Failed servoing attempt");
      if (servo_attempts == 0) {
        servo_success = true;
        ROS_INFO_STREAM("Servoing unsuccessful");
      }
    }
    reg_complete = true;
  }

  void readImageColor(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image)
  {
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, "bgr8");
    pCvImage->image.copyTo(image);
    // cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
  }

  void readImageGrey(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image)
  {
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, "mono8");
    pCvImage->image.copyTo(image);
  }

  void mouse_click(int event, int x, int y)
  {

    switch(event)
    {
      case cv::EVENT_LBUTTONDOWN:
      {

        pointA.x=x;
        pointA.y=y;
        break;
      }

      case cv::EVENT_LBUTTONUP:
      {

        pointB.x=x;
        pointB.y=y;

        displaySelectedRegion();

        reset_pixl = true;

        watsonScale = objectposx - move_group.getCurrentPose().pose.position.x - 0.17;
        multiplier = float(color.cols) / regionOfInterest.cols;
        multiplier2 = float(color.rows) / regionOfInterest.rows;
        // multiplier = multiplier ? color.rows / regionOfInterest.rows < multiplier : color.rows / regionOfInterest.rows;
        watsonScale /= multiplier;
        break;
      }

      case cv::EVENT_MBUTTONDOWN:
      {

        pointA.x = 0;
        pointA.y = 0;
        pointB.x = color.size().width;
        pointB.y = color.size().height;

        displaySelectedRegion();

        watsonScale = objectposx - move_group.getCurrentPose().pose.position.x - 0.17;

        reset_pixl = true;
        break;
      }

      case cv::EVENT_RBUTTONDOWN:
      {
        new_pixl();
        new_watson();
        break;
      }
    }

  }

  static void mouse_click(int event, int x, int y, int, void* this_) {
    static_cast<pixl*>(this_)->mouse_click(event, x, y);
  }

  void displaySelectedRegion() {

    servoing = true;
    time(&timer);

    time_id = difftime(timer,0);

    color(cv::Rect(pointA.x, pointA.y, pointB.x-pointA.x, pointB.y-pointA.y)).copyTo(regionOfInterest);
    color.copyTo(watson_initial_image);
    cv::rectangle(watson_initial_image, pointA, pointB, cv::Scalar(255,255,0), 2);
    cv::namedWindow(REGION_WINDOW,cv::WINDOW_NORMAL);
    // cv::setMouseCallback(REGION_WINDOW, feature_click, this);
    cv::imshow(REGION_WINDOW, regionOfInterest);
    watsonMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", regionOfInterest).toImageMsg();
    watsonPub.publish(watsonMsg);

    cv::namedWindow(LO_RES_WINDOW,cv::WINDOW_NORMAL);
    imageLRSub = nh_.subscribe("/camera_lores/image", 10, &pixl::callbackLR, this);
    cv::setMouseCallback(LO_RES_WINDOW, pixl_click, this);
  }

  void new_estimate() {

    objectposx = create_random(1.0,0.05);

  }

  void feature_click(int event, int x, int y)
  {
    switch(event)
    {
      case cv::EVENT_LBUTTONDOWN:
      {

        featurePoint.x = x;
        featurePoint.y = y;
        displayFeatureRegion();

        break;
      }
    }

  }

  static void feature_click(int event, int x, int y, int, void* this_) {
    static_cast<pixl*>(this_)->feature_click(event, x, y);
  }

  void pixl_click(int event, int x, int y)
  {
    switch(event)
    {
      case cv::EVENT_LBUTTONDOWN:
      {
        pixlMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grey).toImageMsg();
        grey.copyTo(pixl_initial_image);
        pixlPub.publish(pixlMsg);
        scalePub.publish(scale);
        pixlScale = objectposx - move_group.getCurrentPose().pose.position.x - 0.17;
        updateSendScale();
        break;
      }

      case cv::EVENT_MBUTTONDOWN:
      {

        reset_pixl = true;
        break;
      }

      case cv::EVENT_RBUTTONDOWN:
      {

        reset_watson = true;

        break;
      }
    }



  }

  static void pixl_click(int event, int x, int y, int, void* this_) {
    static_cast<pixl*>(this_)->pixl_click(event, x, y);
  }

  void displayFeatureRegion() {

    cv::Point rad(20,20);

    regionOfInterest(cv::Rect(featurePoint-rad,featurePoint+rad)).copyTo(featureOfInterest);

    cv::namedWindow(FEATURE_WINDOW,cv::WINDOW_NORMAL);
    cv::imshow(FEATURE_WINDOW, featureOfInterest);
    cv::namedWindow(LO_RES_WINDOW,cv::WINDOW_NORMAL);
    imageLRSub = nh_.subscribe("/camera_lores/image", 10, &pixl::callbackLR, this);
    cv::setMouseCallback(LO_RES_WINDOW, pixl_click, this);

  }

  void save_images() {
    std::ostringstream s;
    s << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_watson_initial.png";
    cv::imwrite(s.str(), watson_initial_image );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_pixl_initial.png";
    cv::imwrite(s.str(), pixl_initial_image );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_regofint.png";
    cv::imwrite(s.str(), regionOfInterest );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_pixl_final.png";
    cv::imwrite(s.str(), grey );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_watson_final.png";
    cv::imwrite(s.str(), color );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_pixl_est.png";
    cv::imwrite(s.str(), pixl_est_image );

  }

  void save_result(){
    result_file.open("/home/james/Dropbox/NASA/experiment/sim_results.csv", std::ios_base::app);
    save_images();
    // unique id
    result_file << dtoi(time_id) << ",";

    // successful?
    bool test = true?operator_success == "y":false;
    result_file << test << ",";

    // record estimate to surface
    result_file << objectposx << ",";

    // watson position
    result_file << watson_pose.pose.position.x << ",";
    result_file << watson_pose.pose.position.y << ",";
    result_file << watson_pose.pose.position.z << ",";
    tf::Quaternion q(watson_pose.pose.orientation.x, watson_pose.pose.orientation.y, watson_pose.pose.orientation.z, watson_pose.pose.orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    result_file << roll << ",";
    result_file << pitch << ",";
    result_file << yaw << ",";

    // watson full image
    result_file << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_watson_initial.png" << ",";

    // PIXL position
    result_file << pixl_pose.pose.position.x << ",";
    result_file << pixl_pose.pose.position.y << ",";
    result_file << pixl_pose.pose.position.z << ",";
    tf::Quaternion q1(pixl_pose.pose.orientation.x, pixl_pose.pose.orientation.y, pixl_pose.pose.orientation.z, pixl_pose.pose.orientation.w);
    tf::Matrix3x3 m1(q1);
    m1.getRPY(roll, pitch, yaw);
    result_file << roll << ",";
    result_file << pitch << ",";
    result_file << yaw << ",";

    // PIXL full image
    result_file << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_pixl_initial.png" << ",";

    // region of interest image
    result_file << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_regofint.png" << ",";

    // estimated position
    result_file << est_pose.pose.position.x << ",";
    result_file << est_pose.pose.position.y << ",";
    result_file << est_pose.pose.position.z << ",";
    tf::Quaternion q2(est_pose.pose.orientation.x, est_pose.pose.orientation.y, est_pose.pose.orientation.z, est_pose.pose.orientation.w);
    tf::Matrix3x3 m2(q2);
    m2.getRPY(roll, pitch, yaw);
    result_file << roll << ",";
    result_file << pitch << ",";
    result_file << yaw << ",";

    // PIXL full image
    result_file << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_pixl_est.png" << ",";

    // servoed position
    result_file << servo_pose.pose.position.x << ",";
    result_file << servo_pose.pose.position.y << ",";
    result_file << servo_pose.pose.position.z << ",";
    tf::Quaternion q3(servo_pose.pose.orientation.x, servo_pose.pose.orientation.y, servo_pose.pose.orientation.z, servo_pose.pose.orientation.w);
    tf::Matrix3x3 m3(q3);
    m3.getRPY(roll, pitch, yaw);
    result_file << roll << ",";
    result_file << pitch << ",";
    result_file << yaw << ",";

    // final PIXL Image
    result_file << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_pixl_final.png" << ",";

    // final Watson image
    result_file << "/home/james/Dropbox/NASA/experiment/sim_images/"<< dtoi(time_id) << "_watson_final.png" << ",";

    // add any notes
    result_file << note << ",";
    note.str("");
    note.clear();

    result_file << std::endl;
    result_file.close();

  }

  int dtoi(double d)
  {
    return ceil(d - 0.5);
  }

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "pixl");
  pixl pxl;

  pxl.start();

  return 0;
}
