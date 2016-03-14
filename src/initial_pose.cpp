// REMEMBER to set ROS_MASTER_URI and ROS_IP, also source harvey_ws/src/.external on UR5
// roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch limited:=true
// roslaunch ur_description ur5_upload.launch limited:=true
// roslaunch ur5_moveit_config moveit_rviz.launch config:=true limited:=true
// rosrun pixl pixl

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
static const std::string LO_RES_WINDOW = "Ref Pos: Right-Click, Sensor Pos: Middle-Click";
static const std::string REGION_WINDOW = "Selected Region Of Interest";
static const std::string FEATURE_WINDOW = "Feature Region";

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <map>
#include <functional>
#include <stdlib.h>
#include "/opt/ros/indigo/include/gazebo_msgs/DeleteModel.h"
#define HALF_HFOV 27               // half horizontal FOV of the camera
#define HALF_VFOV 18               // half vertical FOV of the camera
#define CAMERA_OFFSET 0.10    // additional offset for camera, not included in model
// #define IM_WIDTH 400          // width of image used for SeqSLAM registration

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
  ros::Publisher recordPub;
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
  float Froi;
  std::vector<moveit_msgs::CollisionObject> collision_objects;

  std::vector<geometry_msgs::Pose> waypoints;

  tf::TransformListener tf_listener;
  float watsonScale;
  float watsonScaleFromRef;
  float watsonScaleFromROI;
  float pixlScale;
  float pixlMult;
  float watsonMult;
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
  float objectposz;
  float objectposzInit;
  float wait_time;
  std::vector<float> servo_data;
  bool reg_complete;
  bool reg_success;
  float servo_scale;
  float servo_rotation;
  float servo_x;
  float servo_y;
  int servo_attempts;
  bool servo_success;
  std::string operator_success;
  std::ofstream result_file;
  time_t timer;
  int time_id;
  bool servoing;
  bool initialise;
  std_msgs::String recordMsg;

  // // servo thresholds - tight
  // float thresh[4] = {0.01,1,5,5};

  // servo thresholds - LOOSE
  float thresh[4] = {0.03,3,15,15};

public:
  pixl()
  : it_(nh_), move_group("manipulator")
  {
    cv::namedWindow(HI_RES_WINDOW,cv::WINDOW_NORMAL);
    cv::namedWindow(LO_RES_WINDOW,cv::WINDOW_NORMAL);
    watsonPub = it_.advertise("/pixl/image_watson", 1);
    pixlPub = it_.advertise("/pixl/image_pixl", 1);
    servoPub = it_.advertise("/pixl/image_servo", 1);
    scalePub = nh_.advertise<std_msgs::Float32MultiArray>("/pixl/scale", 1);
    recordPub = nh_.advertise<std_msgs::String>("/pixl/record", 1);

    imageHRSub = nh_.subscribe("/usb_cam/image_raw", 10, &pixl::callbackHR, this);
    matlabSub = nh_.subscribe("/pixl/matlab_result", 10, &pixl::matlabCallback, this);
    servoSub = nh_.subscribe("/pixl/servo_result", 10, &pixl::servoCallback, this);
    watsonScale = 1.0;
    pixlScale = 1.0;
    servoing = false;

    note.str("");

    // object at 0 in world-z, gaussian noise
    new_estimate();

    ROS_INFO_STREAM("Object estimated at: " << objectposz << " meters.");

    ROS_INFO_STREAM("Init complete");

  }

  ~pixl()
  {

    std::vector<std::string> object_names = planning_scene_interface.getKnownObjectNames();
    planning_scene_interface.removeCollisionObjects(object_names);
    cv::destroyWindow(HI_RES_WINDOW);
    cv::destroyWindow(LO_RES_WINDOW);
    cv::destroyWindow(REGION_WINDOW);
    // cv::destroyWindow(FEATURE_WINDOW);

  }

  void start(){

    std::string xInput;
    std::string yInput;
    std::string zInput;

    wait_time = 1.0;
    cv::setMouseCallback(HI_RES_WINDOW, mouse_click, this);
    cv::setMouseCallback(LO_RES_WINDOW, pixl_click, this);

    reset_pixl = false;
    ros::Time time;

    move_group.setPlannerId("RRTConnectkConfigDefault");
    move_group.setPlanningTime(5);
    move_group.setMaxVelocityScalingFactor(0.25);
    move_group.setPoseReferenceFrame("world");
    ros::AsyncSpinner spinner(2);
    spinner.start();
    initialise = true;

    ROS_INFO_STREAM("Start complete");

    while(ros::ok()){

      if(initialise){

        new_pixl();

        // try {
        //   if(!moveTo(pixl_pose,2, wait_time)){throw tf::TransformException("Pixl pose initial failed.");}
        // } catch (tf::TransformException ex){
        //   ROS_INFO_STREAM(ex.what());
        // }

        new_watson();

        try {
          if(!moveTo(watson_pose,2, wait_time)){throw tf::TransformException("Watson pose initial failed.");}
        } catch (tf::TransformException ex){
          ROS_INFO_STREAM(ex.what());
        }

        reset_pixl = false;
        reset_watson = false;
        initialise = false;

      }

      else if(reset_pixl){

        reset_pixl = false;
        try {
          if(!moveTo(pixl_pose,2, wait_time)){throw tf::TransformException("Pixl pose reset failed.");}
        } catch (tf::TransformException ex){
          ROS_INFO_STREAM(ex.what());
        }

      }

      else if (reset_watson) {

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

  void new_pixl() {
    pixl_pose.pose.position.x = create_random(0.0,0.1);
    pixl_pose.pose.position.y = create_random(-0.6,0.05);
    pixl_pose.pose.position.z = create_random(0.7,0.15);
    pixl_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, M_PI/2, M_PI/2);

    ROS_INFO("\n\nNew PIXL initial position\n\tX: %0.3f\n\tY: %0.3f\n\tZ: %0.3f\n", pixl_pose.pose.position.x, pixl_pose.pose.position.y, pixl_pose.pose.position.z);
    // reset_pixl = true;
  }

  void new_watson() {

    watson_pose.pose.position.x = create_random(0.0,0.05);
    watson_pose.pose.position.y = create_random(-0.6,0.05);
    watson_pose.pose.position.z = create_random(1.5,0.05);
    watson_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, M_PI/2, M_PI/2);

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
      pixlMult = 1;
      watsonMult = watsonScale / pixlScale;

    } else {
      pixlMult = pixlScale / watsonScale;
      watsonMult = 1;
    }
    scale.data.push_back(watsonMult);
    scale.data.push_back(pixlMult);

    scalePub.publish(scale);

  }

  // bool moveToNamed(std::string namedGoal, int Attempts, float timeout){
  //
  //   bool ret = false;
  //   bool success = false;
  //
  //   while(Attempts > 0){
  //
  //     move_group.setStartStateToCurrentState();
  //     move_group.setNamedTarget(namedGoal);
  //
  //     success = move_group.plan(plan);
  //
  //     if(success){
  //
  //       try{
  //
  //         move_group.move();
  //         Attempts = 0;
  //         ret = true;
  //
  //       }
  //
  //       catch(moveit::planning_interface::MoveItErrorCode ex) {
  //
  //         std::cout << "Something went wrong. Failed to move to Goal" << std::endl;
  //         ret = false;
  //
  //       }
  //
  //     } else {
  //
  //       std::cout << "Planning unsuccessful" << std::endl;
  //       Attempts--;
  //       sleep(timeout);
  //
  //     }
  //
  //   }
  //
  //   return ret;
  //
  // }

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

  // void calculateTrajectory(int numSteps){
  //
  //   geometry_msgs::Pose current = move_group.getCurrentPose().pose;
  //
  //   float dx = (arm_pose.pose.position.x - current.position.x) / numSteps;
  //   float dy = (arm_pose.pose.position.y - current.position.y) / numSteps;
  //   float dz = (arm_pose.pose.position.z - current.position.z) / numSteps;
  //
  //   for (int i=0;i<numSteps;i++){
  //
  //     current.position.x += dx;
  //     current.position.y += dy;
  //     current.position.z += dz;
  //
  //     waypoints.push_back(current);
  //
  //   };
  //
  //   waypoints.begin();
  // }
  //
  // bool moveToCartesianPath(geometry_msgs::PoseStamped targetPose, int Attempts, float timeout, bool async){
  //
  //   bool response = true;
  //   ros::ServiceClient executeKnownTrajectoryServiceClient = nh_.serviceClient<moveit_msgs::ExecuteKnownTrajectory>("/execute_kinematic_path");
  //
  //   move_group.setPoseReferenceFrame("world");
  //   move_group.setStartStateToCurrentState();
  //   move_group.setPoseTarget(targetPose);
  //
  //   // set waypoints for which to compute path
  //   std::vector<geometry_msgs::Pose> waypoints;
  //   waypoints.push_back(move_group.getCurrentPose().pose);
  //   waypoints.push_back(targetPose.pose);
  //   moveit_msgs::ExecuteKnownTrajectory srv;
  //
  //   // compute cartesian path
  //   double ret = move_group.computeCartesianPath(waypoints, 0.1, 10000, srv.request.trajectory, false);
  //   if(ret < 0){
  //     // no path could be computed
  //     ROS_INFO_STREAM("Unable to compute Cartesian path!");
  //     response = false;
  //   } else if (ret < 1){
  //     // path started to be computed, but did not finish
  //     ROS_INFO_STREAM("Cartesian path computation finished " << ret * 100 << "% only!");
  //     response = false;
  //   }
  //
  //   // send trajectory to arm controller
  //   srv.request.wait_for_execution = true;
  //   executeKnownTrajectoryServiceClient.call(srv);
  //   return response;
  // }

  // void addImageNoise(cv::Mat &image) {
  //   cv::Mat noise = cv::Mat::zeros(image.size(),CV_8U);
  //   cv::randn(noise, cv::Scalar(0), cv::Scalar(20));
  //   cv::add(image,noise,image);
  //
  // }

  void callbackHR(const sensor_msgs::Image::ConstPtr imageColor)
  {
    readImageColor(imageColor, color);
    // addImageNoise(color);
    cv::imshow(HI_RES_WINDOW, color);
    cv::waitKey(3);
  }

  // void callbackLR(const sensor_msgs::Image::ConstPtr imageGrey)
  // {
  //   readImageGrey(imageGrey, grey);
  //   cv::imshow(LO_RES_WINDOW, grey);
  //   cv::waitKey(3);
  // }

  void matlabCallback(const std_msgs::Float32MultiArray msg)
  {
    //set servo flag
    servoing = true;
    ROS_INFO_STREAM("SeqSLAM information received from MATLAB");

    //store scale, rot, Tx, Ty in usable form
    // ROS_INFO_STREAM(msg.data);
    std::vector<float> data;
    data = msg.data;
    Eigen::Map<Eigen::MatrixXf> mat(data.data(), msg.layout.dim[0].size, msg.layout.dim[1].size);
    // ROS_INFO_STREAM(mat);
    // if the initial registration fails, print message, cancel servoing, make note
    if (mat(0,0) == 0) {
      ROS_INFO_STREAM("Initial registration failed.");
      servoing = false;
      note << "Initial registration failed. ";
      return;
    }

    ROS_INFO_STREAM("MATLAB returned: " << mat);
    // move to estimated extension
    arm_pose = move_group.getCurrentPose();
    ROS_INFO_STREAM("pixlScale: " << pixlScale);
    // ROS_INFO_STREAM("obj estimate: " << objectposz);
    // objectposz = objectposz + (watsonScale - watsonScale*(1 + (mat(0,0)-1)*watsonMult/pixlMult)) * (float(color.cols) / regionOfInterest.cols);
    // ROS_INFO_STREAM("obj estimate updated: " << objectposz);
    ROS_INFO_STREAM("watsonScale: " << watsonScale/pixlMult);
    watsonScale /= mat(0,0); //(1 + (mat(0,0)-1)*watsonMult);
    ROS_INFO_STREAM("watsonScale updated: " << watsonScale);
    // arm_pose.pose.position.z = - objectposz + watsonScale + CAMERA_OFFSET; // estimate vertical position based on returned scale
    arm_pose.pose.position.z = watsonScale + CAMERA_OFFSET; // estimate vertical position based on returned scale
    ROS_INFO_STREAM("z: " << arm_pose.pose.position.z);
    float dx = (watsonScale * tan(HALF_HFOV*M_PI/180) * 2) * (mat(0,3) / color.cols); // (estimate of width of image (width) contents) * (translation in pixels / image width in pixels)
    float dy = (watsonScale * tan(HALF_HFOV*M_PI/180) * 2) * (mat(0,2) / color.cols);
    ROS_INFO_STREAM("dx: " << dx);
    ROS_INFO_STREAM("dy: " << dy);
    arm_pose.pose.position.x -= dx;
    arm_pose.pose.position.y += dy; // sign due to world axes

    ROS_INFO_STREAM("Moving to estimated pose: \n" << arm_pose);

    try{

      if(!moveTo(arm_pose,2, wait_time)){throw tf::TransformException("Move to estimated position failed");}

    }
    catch (tf::TransformException ex){

      // if move unsuccessful, reset to pixl then watson positions, record to note,
      note << ex.what();
      ROS_INFO_STREAM(ex.what());
      servoing = false;

    }
    sleep(wait_time); //avoid vibration in image
    if (!servoing) {
      ROS_INFO_STREAM("Not servoing");
    } else {
      est_pose = move_group.getCurrentPose();
      color.copyTo(pixl_est_image);
      // closed_servo(); // PID-controlled close loop servoing
      open_servo(); // OPEN LOOP "SERVOING"
    }

    std::cout << "Successful attempt [Y/n]? ";
    std::cin >> operator_success;
    servo_pose = move_group.getCurrentPose();
    save_result();
    servoing = false;

    new_pixl();
    reset_pixl = false;
    new_watson();
    new_estimate();
    // reset_watson = true;
  }

  void closed_servo() {

    ROS_INFO_STREAM("Begin closed-loop servoing");
    init_scale();
    scalePub.publish(scale);
    servo_attempts = 10;
    servo_success = false;
    reg_success = false;
    std::vector<float> temp(4,0);
    Eigen::ArrayXXf set_point(1,4);
    set_point << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf err_sum(1,4);
    err_sum << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf d_err(1,4);
    d_err << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf lastErr(1,4);
    lastErr << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf current(1,4);
    current << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf error(1,4);
    error << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf output(1,4);
    output << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf kp(1,4);
    kp << 0.05, 0.5, 0.00005, 0.00005;
    Eigen::ArrayXXf ki(1,4);
    ki << 0.0, 0.0, 0.0, 0.0;
    Eigen::ArrayXXf kd(1,4);
    kd << 0.0, 0.0, 0.00001, 0.00001;

    while(!servo_success) {

      // servoMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grey).toImageMsg();
      servoMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color).toImageMsg();
      servoPub.publish(servoMsg);
      reg_complete = false;
      while(!reg_complete) {

      }

      if (reg_success) {

        arm_pose = move_group.getCurrentPose();

        // Compute all the working error variables
        current << 1-servo_scale,-servo_rotation,servo_x,servo_y;

        error = set_point - current;

        err_sum += error;
        d_err = (error - lastErr);

        // /*Compute PID Output*/
        output = kp * error + ki * err_sum + kd * d_err;

        tf::Quaternion q(arm_pose.pose.orientation.x, arm_pose.pose.orientation.y, arm_pose.pose.orientation.z, arm_pose.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        arm_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw + output(0,1));
        arm_pose.pose.position.z -= output(0,0);
        arm_pose.pose.position.x -= output(0,2);
        arm_pose.pose.position.y -= output(0,3);

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

        sleep(wait_time);

      }

    }

  }

  void open_servo() {

    ROS_INFO_STREAM("Begin open-loop servoing");

    servo_success = false;

    while(!servo_success) {

      arm_pose = move_group.getCurrentPose();
      // pixlScale = arm_pose.pose.position.z - CAMERA_OFFSET - objectposz;
      // ROS_INFO_STREAM("pixlScale: " << pixlScale);
      // updateSendScale();

      // servoMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grey).toImageMsg();
      servoMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color).toImageMsg();
      init_scale();
      scalePub.publish(scale);
      servoPub.publish(servoMsg);
      reg_complete = false;

      while(!reg_complete) {

      }

      if (reg_success) {
        tf::Quaternion q(arm_pose.pose.orientation.x, arm_pose.pose.orientation.y, arm_pose.pose.orientation.z, arm_pose.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        arm_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw - servo_rotation);
        // ROS_INFO_STREAM("obj estimate: " << objectposz);
        // objectposz = objectposz + (watsonScale - watsonScale*(1 + (servo_scale-1)*watsonMult/pixlMult)) * (float(color.cols) / regionOfInterest.cols);
        // ROS_INFO_STREAM("obj estimate updated: " << objectposz);
        ROS_INFO_STREAM("watsonScale: " << watsonScale);
        watsonScale /= servo_scale; //(1 + (servo_scale-1)*watsonMult/pixlMult);
        ROS_INFO_STREAM("watsonScale updated: " << watsonScale);
        // arm_pose.pose.position.z = - objectposz + watsonScale + CAMERA_OFFSET; // (estimate of object position in world)
        arm_pose.pose.position.z = watsonScale + CAMERA_OFFSET; // (estimate of object position in world)
        ROS_INFO_STREAM("z: " << arm_pose.pose.position.z);
        float dx = (watsonScale * tan(HALF_HFOV*M_PI/180) * 2) * (-servo_x / color.cols); // (estimate of width of image (width) contents) * (translation in pixels / image width in pixels)
        float dy = (watsonScale * tan(HALF_HFOV*M_PI/180) * 2) * (servo_y / color.cols);
        ROS_INFO_STREAM("dx: " << dx);
        ROS_INFO_STREAM("dy: " << dy);
        arm_pose.pose.position.x -= dx;
        arm_pose.pose.position.y += dy; // sign due to world axes


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

        sleep(wait_time);
      }

    }

  }

  void servoCallback(const std_msgs::Float32MultiArray msg)
  {

    ROS_INFO_STREAM("SURF information received from MATLAB");
    servo_data = msg.data;
    Eigen::Map<Eigen::MatrixXf> mat(servo_data.data(), msg.layout.dim[0].size, msg.layout.dim[1].size);
    reg_success = true ? mat(0,0) : false;
    if (reg_success) {
      ROS_INFO_STREAM("MATLAB returned: " << mat);
      servo_scale = mat(0,1);
      servo_rotation = mat(0,2);
      servo_x = -mat(0,4);
      servo_y = mat(0,3);
      servo_attempts = 10;
      if (abs(1-servo_scale) < thresh[0] && abs(servo_rotation) < thresh[1] * M_PI/180 && abs(servo_x) < thresh[2] && abs(servo_y) < thresh[3]) {
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

  // void readImageGrey(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image)
  // {
  //   cv_bridge::CvImageConstPtr pCvImage;
  //   pCvImage = cv_bridge::toCvShare(msgImage, "mono8");
  //   pCvImage->image.copyTo(image);
  // }

  void mouse_click(int event, int x, int y)
  {

    switch(event)
    {
      case cv::EVENT_LBUTTONDOWN:
      {

        pointA.x=x;
        pointA.y=y;
        time(&timer);

        time_id = dtoi(difftime(timer,0));
        recordMsg.data = std::to_string(time_id);
        recordPub.publish(recordMsg);
        break;
      }

      case cv::EVENT_LBUTTONUP:
      {

        pointB.x=x;
        pointB.y=y;

        displaySelectedRegion();

        reset_pixl = true;

        watsonScale = move_group.getCurrentPose().pose.position.z - CAMERA_OFFSET - objectposz;
        watsonScaleFromRef = watsonScale;
        Froi = float(color.cols) / regionOfInterest.cols;
        watsonScale /= Froi;
        watsonScaleFromROI = watsonScale;

        break;
      }

      // case cv::EVENT_MBUTTONDOWN:
      // {
      //
      //   pointA.x = 0;
      //   pointA.y = 0;
      //   pointB.x = color.size().width;
      //   pointB.y = color.size().height;
      //
      //   displaySelectedRegion();
      //
      //   watsonScale = move_group.getCurrentPose().pose.position.z - CAMERA_OFFSET - objectposz;
      //
      //   reset_pixl = true;
      //   break;
      // }

      case cv::EVENT_RBUTTONDOWN:
      {
        // new_pixl();
        // new_watson();
        // break;

        pixlMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color).toImageMsg();
        color.copyTo(pixl_initial_image);
        pixlPub.publish(pixlMsg);
        scalePub.publish(scale);
        pixlScale = move_group.getCurrentPose().pose.position.z - CAMERA_OFFSET - objectposz;
        updateSendScale();
        break;
      }
    }

  }

  static void mouse_click(int event, int x, int y, int, void* this_) {
    static_cast<pixl*>(this_)->mouse_click(event, x, y);
  }

  void displaySelectedRegion() {

    servoing = true;

    color(cv::Rect(pointA.x, pointA.y, pointB.x-pointA.x, pointB.y-pointA.y)).copyTo(regionOfInterest);
    color.copyTo(watson_initial_image);
    cv::rectangle(watson_initial_image, pointA, pointB, cv::Scalar(255,255,0), 2);
    cv::namedWindow(REGION_WINDOW,cv::WINDOW_NORMAL);
    // cv::setMouseCallback(REGION_WINDOW, feature_click, this);
    cv::imshow(REGION_WINDOW, regionOfInterest);
    watsonMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", regionOfInterest).toImageMsg();
    watsonPub.publish(watsonMsg);

    // cv::namedWindow(LO_RES_WINDOW,cv::WINDOW_NORMAL);
    // imageLRSub = nh_.subscribe("/camera_lores/image", 10, &pixl::callbackLR, this);
    // cv::setMouseCallback(LO_RES_WINDOW, pixl_click, this);
  }

  void new_estimate() {

    objectposz = create_random(0.0,0.1);
    // objectposz = 0.05;
    objectposzInit = objectposz;

  }

  // void feature_click(int event, int x, int y)
  // {
  //   switch(event)
  //   {
  //     case cv::EVENT_LBUTTONDOWN:
  //     {
  //
  //       featurePoint.x = x;
  //       featurePoint.y = y;
  //       displayFeatureRegion();
  //
  //       break;
  //     }
  //   }
  //
  // }
  //
  // static void feature_click(int event, int x, int y, int, void* this_) {
  //   static_cast<pixl*>(this_)->feature_click(event, x, y);
  // }

  void pixl_click(int event, int x, int y)
  {
    switch(event)
    {
      // case cv::EVENT_LBUTTONDOWN:
      // {
      //   pixlMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grey).toImageMsg();
      //   grey.copyTo(pixl_initial_image);
      //   pixlPub.publish(pixlMsg);
      //   scalePub.publish(scale);
      //   pixlScale = objectposz + move_group.getCurrentPose().pose.position.z - CAMERA_OFFSET;
      //   HALF_HFOVdScale();
      //   break;
      // }

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
    // imageLRSub = nh_.subscribe("/camera_lores/image", 10, &pixl::callbackLR, this);
    // imageLRSub = nh_.subscribe("/nothing", 10, &pixl::callbackLR, this);
    cv::setMouseCallback(LO_RES_WINDOW, pixl_click, this); // used for user input for moving between watson and pixl positions

  }

  void save_images() {
    std::ostringstream s;
    s << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_ref_initial.png";
    cv::imwrite(s.str(), watson_initial_image );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_sensor_initial.png";
    cv::imwrite(s.str(), pixl_initial_image );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_regofint.png";
    cv::imwrite(s.str(), regionOfInterest );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_sensor_final.png";
    cv::imwrite(s.str(), color );
    s.str("");
    s.clear();
    s << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_sensor_est.png";
    cv::imwrite(s.str(), pixl_est_image );

  }

  void save_result(){
    result_file.open("/home/james/Dropbox/NASA/experiment/robust_results.csv", std::ios_base::app);
    save_images();
    // unique id
    result_file << time_id << ",";

    // successful?
    bool test = true?operator_success == "y":false;
    result_file << test << ",";

    // record estimate to surface
    result_file << objectposzInit << ",";

    // record estimate of distance to sample surface
    result_file << watsonScaleFromRef << ",";

    // record estimate of distance to sample surface
    result_file << watsonScaleFromROI << ",";

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
    result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_ref_initial.png" << ",";

    // region of interest image
    result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_regofint.png" << ",";

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
    result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_sensor_initial.png" << ",";

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
    result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_sensor_est.png" << ",";

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

    // final image
    result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"<< time_id << "_sensor_final.png" << ",";

    // record estimate of distance to sample surface
    result_file << watsonScale << ",";

    // add any notes
    result_file << note.str() << ",";
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
