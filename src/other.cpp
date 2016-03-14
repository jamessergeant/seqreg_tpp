#include <ros/ros.h>
#include <ros/time.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <visualization_msgs/Marker.h>

#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64.h>

#include <ur_msgs/SetIO.h>
#include <ur_msgs/IOStates.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <string>
#include <vector>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>

#include "superquadric_fitter/estimateCapsicumPose.h"
#include "capsicum_detection/segmentCapsicum.h"
#include <capsicum_detector.h>
#include <cloud_filtering_tools.h>

static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW2 = "Image window2";
static const std::string OPENCV_WINDOW3 = "Image window3";

static const double MSG_PULSE_SEC = 0.1;
static const double WAIT_GRIPPER_CLOSE_SEC = 0.1;
static const double WAIT_STATE_MSG_SEC = 1; // max time to wait for the gripper state to refresh
static const double GRIPPER_MSG_RESEND = 10; // Number of times to re-send a msg to the end effects for assurance that it arrives

#define DEBUG

class harvey
{

    enum ROBOT_STATE_FLAG {
        RESET,
        START,
        DETECT_INIT_FRUIT,
        SCANNING,
        DETECT_FRUIT,
        ATTACH,
        CUT,
        PLACE
    };

  std::vector<std::string> ROBOT_STATES_NAMES = {
      "RESET",
      "START",
      "DETECT_INIT_FRUIT",
      "SCANNING",
      "DETECT_FRUIT",
      "ATTACH",
      "CUT",
      "PLACE"
  };


  //capsicum detector code
  HSV_model capsicum_hsv_model;
  capsicum_detector capsicumDetector;

  geometry_msgs::PointStamped target_capsicum_centroid;
  std::vector<capsicum> capsicums;



  std::vector<moveit_msgs::CollisionObject> collision_objects;

  tf::TransformListener tf_listener;

  bool pick_fruit, found_fruit, reset_flag;
  int  pick_fruit_index;

  int SUCTION_PIN = 4;
  int CUTTER_PIN = 5;
  int PRESSURE_SWITCH = 0;


  cv::Mat cameraMatrixColor;
  cv::Mat cameraMatrixDepth;
  cv::Mat scene_image;

  cv::Mat lookupX, lookupY;

  geometry_msgs::PoseStamped capsicum_pose;
  geometry_msgs::PoseStamped capsicum_base_pose;
  geometry_msgs::PoseStamped arm_pose;
  Eigen::Affine3d capsicum_pose_eigen;
  Eigen::Vector3d capsicum_model;

  sensor_msgs::PointCloud2 scene_cloud_msg;

  std::vector<double> start_joint_values = {0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0};

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

  image_transport::Subscriber image_sub;

  ros::Subscriber capsicum_cloud_sub, scene_cloud_sub, io_states_sub, scene_image_sub;
  ros::Publisher wire_cutter_pub, vis_pub, kinfu_reset_pub, kinfu_pause_pub, estimate_capsicum_pub;

  ros::ServiceClient io_client;
  ros::ServiceClient estimateCapsicumPose_client;
  ros::ServiceClient segmentCapsicum_client;

  std::string gripperPoseTopic, kinfuResetTopic, kinfuPauseTopic, superquadricTopic, urIoTopic, urIoStateTopic;
  std::string segmentCapsicumTopic, topicCapsicumCloud, topicCapsicumImage, topicKinfuSceneCloud;

  ROBOT_STATE_FLAG robot_state_;

  Eigen::Affine3d grasp_pose_eigen;

  moveit::planning_interface::MoveGroup::Plan ur5_plan;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  moveit::planning_interface::MoveGroup ur5;
  double wait_time;
  int attempts;

  std::vector<ur_msgs::Digital> digital_in_states;

  //Starting Position
  Eigen::Vector3d delta_xyz, tool_point, scan_offset, cutting_offset;
  Eigen::Vector4d start_q;
  double velocity_scan, velocity_attach, velocity_cut,gripper_offset_up,gripper_offset_in, gripper_offset_init, gripper_offset_side;
  double row_depth, focal_x, focal_y;
  double tool_point_x, tool_point_y, tool_point_z, tool_offset_x;
//  double cutting_offset_z, cutting_offset_x;
  double eef_step_size, jump_threshold;

public:
  harvey()
    : nh_("~"),it_(nh_), ur5("manipulator")
    {
        cv::namedWindow(OPENCV_WINDOW);

        nh_.param("gripper_pose_topic",gripperPoseTopic,std::string("/gripper_pose"));
        nh_.param("kinfu_reset_topic",kinfuResetTopic,std::string("/ros_kinfu/reset"));
        nh_.param("kinfu_pause_topic",kinfuPauseTopic,std::string("/ros_kinfu/pause"));
        nh_.param("superquadric_topic",superquadricTopic,std::string("/superquadric_fitter/estimate_capsicum_pose"));
        nh_.param("ur_set_io_topic",urIoTopic,std::string("/set_io"));
        nh_.param("segment_capsicum_topic",segmentCapsicumTopic,std::string("/capsicum_detection/segment_capsicum"));

        //vis_pub = nh_.advertise<geometry_msgs::PoseStamped>( gripperPoseTopic, 0 );
        vis_pub = nh_.advertise<geometry_msgs::PoseArray>( gripperPoseTopic, 0 );


        kinfu_reset_pub = nh_.advertise<std_msgs::Empty>( kinfuResetTopic, 0 );
        kinfu_pause_pub = nh_.advertise<std_msgs::Empty>( kinfuPauseTopic, 0 );
        wire_cutter_pub = nh_.advertise<std_msgs::Float64>("/wire_cutter_controller1/command", 0);
        io_client = nh_.serviceClient<ur_msgs::SetIO>(urIoTopic);
        estimateCapsicumPose_client = nh_.serviceClient<superquadric_fitter::estimateCapsicumPose>(superquadricTopic);
        segmentCapsicum_client = nh_.serviceClient<capsicum_detection::segmentCapsicum>(segmentCapsicumTopic);

        pick_fruit = false;
        pick_fruit_index = 0;
        found_fruit = false;

        nh_.param("velocity_scan", velocity_scan, 0.1);
        nh_.param("velocity_attach", velocity_attach,0.25);
        nh_.param("velocity_cut", velocity_cut,0.05);

        nh_.param("eef_step_size", eef_step_size,0.01);
        nh_.param("jump_threshold", jump_threshold,0.0);

        //nh_.param("scan_offset",scan_offset, 0.45);
        nh_.param("row_depth",row_depth,0.4);
        nh_.param("focal_x",focal_x, 1340.4504);
        nh_.param("focal_y",focal_y, 1343.7678);

        nh_.param("gripper_offset_x",tool_point[0],0.27);
        nh_.param("gripper_offset_y",tool_point[1],0.012);
//        nh_.param("gripper_offset_y",tool_point[1],0.024);
        nh_.param("gripper_offset_z",tool_point[2],0.075);

        nh_.param("tool_offset_x",tool_offset_x,0.1);

        nh_.param("scan_offset_x",scan_offset[0], 0.35);
        nh_.param("scan_offset_y",scan_offset[1], 0.0);
        nh_.param("scan_offset_z",scan_offset[2], 0.08);

//        nh_.param("scan_offset_x",scan_offset[0], 0.35);
//        nh_.param("scan_offset_y",scan_offset[1], 0.0);
//        nh_.param("scan_offset_z",scan_offset[2], 0.08);

        //double delta_z, delta_x, delta_y;
        nh_.param("scan_delta_x", delta_xyz[0], 0.0);
        nh_.param("scan_delta_y", delta_xyz[1], 0.15);
        nh_.param("scan_delta_z", delta_xyz[2], 0.15);

        nh_.param("start_q1",start_q[0],1.0);
        nh_.param("start_q2",start_q[1],0.0);
        nh_.param("start_q3",start_q[2],0.0);
        nh_.param("start_q4",start_q[3],0.0);

        nh_.param("cutting_offset_x",cutting_offset[0],0.095);
        nh_.param("cutting_offset_y",cutting_offset[1],0.00);
        nh_.param("cutting_offset_z",cutting_offset[2],-0.01);

        nh_.param("wait_time",wait_time,0.0);
        nh_.param("planning_attempts",attempts,4);

        nh_.param("capsicum_cloud_topic",topicCapsicumCloud,std::string("/capsicum/points"));
        nh_.param("kinfu_cloud_topic",topicKinfuSceneCloud,std::string("/ros_kinfu/depth_registered/points"));
        nh_.param("capsicum_image_topic",topicCapsicumImage,std::string("/camera/rgb/image_raw"));
//        nh_.param("capsicum_image_topic",topicCapsicumImage,std::string("/camera/rgb/image_raw_"));

        ROS_INFO_STREAM("Kinfu Cloud Topic: " << topicKinfuSceneCloud);
        ROS_INFO_STREAM("Capsicum Image Topic: " << topicCapsicumImage);

        scene_cloud_sub = nh_.subscribe(topicKinfuSceneCloud, 1, &harvey::scene_cloud_callback, this);
        scene_image_sub = nh_.subscribe(topicCapsicumImage, 1, &harvey::image_callback, this);



        nh_.param("ur_io_states_topic",urIoStateTopic,std::string("/io_states"));
        io_states_sub = nh_.subscribe(urIoStateTopic, 1, &harvey::io_states_callback, this);

        capsicum_hsv_model.hue_mean = 84;
        capsicum_hsv_model.saturation_mean = 221;
        capsicum_hsv_model.value_mean = 121;
        capsicum_hsv_model.hue_var = 20.2;
        capsicum_hsv_model.saturation_var = 1368;
        capsicum_hsv_model.value_var = 753;

        capsicumDetector.capsicum_model = capsicum_hsv_model;

    }

    ~harvey()
    {
        turnOffIO(SUCTION_PIN);
        cv::destroyWindow(OPENCV_WINDOW);
//        std::vector<std::string> object_names = planning_scene_interface.getKnownObjectNames();
//        planning_scene_interface.removeCollisionObjects(object_names);
    }

    void start(){


        cv::setMouseCallback(OPENCV_WINDOW, mouse_click, this);

        reset_flag = false;
        ros::Time time;

        initUR5();

        updateState(START);
        //updateState(DETECT_FRUIT);

        ros::AsyncSpinner spinner(2);
        spinner.start();

        //std::cout << "Moving to neutral" << std::endl;
        //moveToNamed("neutral",2,wait_time);

        kinfu_reset_pub.publish(std_msgs::Empty());

        while(ros::ok())
        {
            sensor_msgs::PointCloud2 segmentedCapsicum;
            switch (robot_state_) {

            case START:


                cout << "Press the ENTER key to pick fruit";
                if (cin.get() == '\n'){

                    ROS_INFO("Starting to Pick fruit");
                    ur5.setMaxVelocityScalingFactor(velocity_scan);

                    if(moveToNamed("start_wide",attempts,wait_time))
                    {
                        //wait for robot to settle before resetting kinfu

                        updateState(DETECT_INIT_FRUIT);
                    }else{
                        updateState(RESET);
                    }
                }

                break;

            case RESET:

                reset();
                break;


            case DETECT_INIT_FRUIT:

                if(detect_init_capsicum_from_image(scene_image, focal_x, focal_y, row_depth, target_capsicum_centroid))
                {
                    if(moveToInitCapsicum(target_capsicum_centroid));
                    {
                        ros::Duration(0.5).sleep();
                        kinfu_reset_pub.publish(std_msgs::Empty());
                        ros::Duration(0.5).sleep();
                        if(detect_init_capsicum(target_capsicum_centroid)){
                            ROS_INFO("Found fruit attempting to scan");
                            updateState(SCANNING);
                        }else{
                            if(moveToNamed("start_wide",attempts,wait_time))
                            {
                                updateState(START);
                            }else{
                                updateState(RESET);
                            }
                        }

                    }
                }
                break;

            case SCANNING:
                if(scanCapsicum(start_q, delta_xyz, scan_offset, target_capsicum_centroid)){
                    updateState(DETECT_FRUIT);
                }else{
                    kinfu_pause_pub.publish(std_msgs::Empty());
                    updateState(RESET);
                }
                break;

            case DETECT_FRUIT:

                if(!segmentCapsicum(scene_cloud_msg,segmentedCapsicum)){
                    updateState(RESET);
                    break;
                }

                if(estimateCapsicum(segmentedCapsicum,capsicum_pose_eigen,capsicum_model)){
                    updateState(ATTACH);
                }else{
                    updateState(RESET);
                }
                break;

            case ATTACH:
                ur5.setMaxVelocityScalingFactor(velocity_attach);
                if(!attachCapsicum(capsicum_pose_eigen,capsicum_model, grasp_pose_eigen)){
                    updateState(RESET);
                }else{
                    updateState(CUT);
                }

                break;

            case CUT:
                ur5.setMaxVelocityScalingFactor(velocity_scan);
                //if(!cutCapsicum(grasp_pose_eigen,capsicum_model)){
                if(!cutCapsicumFlat(capsicum_pose_eigen,capsicum_model)){
                    updateState(RESET);
                }else{
                    updateState(PLACE);
                }

                break;

            case PLACE:
                ur5.setMaxVelocityScalingFactor(velocity_attach);
                //std::vector<geometry_msgs::Pose> waypoints = {start_pose};

                if(moveToNamed("start_wide",attempts,wait_time)){
                    moveToNamed("safe_pose",attempts,wait_time);
                    //moveToNamed("box_field",attempts,wait_time);
                    //moveToNamed("box_",attempts,wait_time);
                    turnOffIO(SUCTION_PIN);
                    //moveToNamed("box_ground",attempts,wait_time);
                    //moveToNamed("safe_pose",attempts,wait_time);
                    moveToNamed("start_wide",attempts,wait_time);

                    found_fruit = false;
                    kinfu_reset_pub.publish(std_msgs::Empty());
                    updateState(START);
                }else{
                    updateState(RESET);
                }

                break;

            }

        }
    }

    void updateState(ROBOT_STATE_FLAG new_state){
        robot_state_ = new_state;
        ROS_INFO_STREAM("Robot State: " << ROBOT_STATES_NAMES[new_state] );
    }

    void initUR5(){
        ur5.setPlannerId("RRTConnectkConfigDefault");

        //ur5.setPlannerId("KPIECEkConfigDefault");

        ur5.setPlanningTime(10);
        ur5.setMaxVelocityScalingFactor(velocity_attach);
        turnOffIO(SUCTION_PIN);
        turnOffIO(CUTTER_PIN);

    }

    void reset(){
        std::cout << "Reset flagged moving to neutral" << std::endl;
        updateState(START);
        reset_flag = false;
        pick_fruit = false;
        found_fruit = false;
        turnOffIO(SUCTION_PIN);
    }

    bool attachCapsicum(Eigen::Affine3d capsicum_pose_eigen, Eigen::Vector3d &capsicum_model, Eigen::Affine3d &grasp_pose_eigen){

        Eigen::Affine3d grasp_pose_translated, tool_point_translation, tool_point_translation2;
        geometry_msgs::Pose grasp_pose;

        std::vector<geometry_msgs::Pose> waypoints;

        geometry_msgs::PoseArray pose_array;
        pose_array.header.frame_id = "/world";
        pose_array.header.stamp = ros::Time::now();


        //before picking change velocity to picking speed
        ur5.setMaxVelocityScalingFactor(velocity_attach);

        //Convert capsicum_pose_eigen to grasp_pose_eigen using axis selection
        //Eigen::Vector3d grasp_offset(gripper_offset_init,gripper_offset_side,gripper_offset_up);
        createGraspTransform(capsicum_pose_eigen, capsicum_model, grasp_pose_eigen, tool_point);
        //tf::poseEigenToMsg(grasp_pose_eigen,grasp_pose);

        // Move to pose offset from capsicum
        tool_point_translation = Eigen::Translation3d(Eigen::Vector3d(-tool_offset_x,0.0,0.0));
        grasp_pose_translated = grasp_pose_eigen*tool_point_translation;

        tf::poseEigenToMsg(grasp_pose_translated,grasp_pose);
        waypoints.push_back(grasp_pose);
        pose_array.poses.push_back(grasp_pose);


        // Translate pose forward to capsicum and move to capsicum
        //tool_point_translation = Eigen::Translation3d(Eigen::Vector3d(tool_offset_x,0,0));
        //grasp_pose_translated = grasp_pose_eigen*tool_point_translation;
        //tf::poseEigenToMsg(grasp_pose_translated,grasp_pose);

        tool_point_translation2 = Eigen::Translation3d(Eigen::Vector3d(0.0075,0.0,0.0));
        grasp_pose_translated = grasp_pose_eigen*tool_point_translation2;

        tf::poseEigenToMsg(grasp_pose_eigen,grasp_pose);
        waypoints.push_back(grasp_pose);
        pose_array.poses.push_back(grasp_pose);

        vis_pub.publish(pose_array);

        // Turn on the suction
        if(!turnOnIO(SUCTION_PIN)) std::cout << "Couldn't Turn on Suction" << std::endl;


        if(!moveToCartesianPath(waypoints,velocity_attach, attempts, wait_time, false, SUCTION_PIN)){return false;}


        //if(!turnOffSuction(4)) std::cout << "Couldn't Turn off  Suction" << std::endl;

        return true;
    }

    bool cutCapsicumFlat(Eigen::Affine3d &grasp_pose_eigen, Eigen::Vector3d &capsicum_model){

          Eigen::Affine3d grasp_translation, grasp_translation1, grasp_translation2, grasp_translation3;
          geometry_msgs::Pose grasp_pose;
          geometry_msgs::PoseArray pose_array;
          pose_array.header.frame_id = "/world";
          pose_array.header.stamp = ros::Time::now();

          tf::poseEigenToMsg(grasp_pose_eigen,grasp_pose);
          grasp_pose.orientation.x = 1;
          grasp_pose.orientation.y = 0;
          grasp_pose.orientation.z = 0;
          grasp_pose.orientation.w = 0;
          tf::poseMsgToEigen(grasp_pose,grasp_pose_eigen);

          std::vector<geometry_msgs::Pose> waypoints1, waypoints2;

          grasp_translation = Eigen::Translation3d(Eigen::Vector3d(-tool_point[0],tool_point[1],-tool_point[2] -capsicum_model[2]/2 -cutting_offset[2] -0.05));
          Eigen::Affine3d grasp_pose_eigen1 = grasp_pose_eigen*grasp_translation;
          tf::poseEigenToMsg(grasp_pose_eigen1,grasp_pose);

          waypoints1.push_back(grasp_pose);
          pose_array.poses.push_back(grasp_pose);

          grasp_translation1 = Eigen::Translation3d(Eigen::Vector3d(-tool_point[0],tool_point[1],-tool_point[2] -capsicum_model[2]/2 -cutting_offset[2]));
          Eigen::Affine3d grasp_pose_eigen2 = grasp_pose_eigen*grasp_translation1;
          tf::poseEigenToMsg(grasp_pose_eigen2,grasp_pose);

          waypoints1.push_back(grasp_pose);
          pose_array.poses.push_back(grasp_pose);

          vis_pub.publish(pose_array);
          pose_array.header.stamp = ros::Time::now();


          if(!moveToCartesianPath(waypoints1,velocity_cut*3.0, attempts, wait_time, false, 0)){return false;}

          if(!turnOnIO(CUTTER_PIN));

          pose_array.poses.clear();

          grasp_translation2 = Eigen::Translation3d(Eigen::Vector3d(-tool_point[0] + cutting_offset[0],tool_point[1],-tool_point[2] -capsicum_model[2]/2 -cutting_offset[2]));
          Eigen::Affine3d grasp_pose_eigen3 = grasp_pose_eigen*grasp_translation2;
          tf::poseEigenToMsg(grasp_pose_eigen3,grasp_pose);
          waypoints2.push_back(grasp_pose);
          pose_array.poses.push_back(grasp_pose);


          grasp_translation3 = Eigen::Translation3d(Eigen::Vector3d(-tool_point[0] -cutting_offset[0],tool_point[1],-tool_point[2] -capsicum_model[2]/2 -cutting_offset[2]));
          Eigen::Affine3d grasp_pose_eigen4 = grasp_pose_eigen*grasp_translation3;
          tf::poseEigenToMsg(grasp_pose_eigen4,grasp_pose);
          waypoints2.push_back(grasp_pose);
          pose_array.poses.push_back(grasp_pose);

          pose_array.header.stamp = ros::Time::now();
          vis_pub.publish(pose_array);
          if(!moveToCartesianPath(waypoints2,velocity_cut, attempts, wait_time, false, 0)){return false;}

           if(!turnOffIO(CUTTER_PIN));
          return true;

      }



    bool cutCapsicum(Eigen::Affine3d &grasp_pose_eigen, Eigen::Vector3d &capsicum_model){

        Eigen::Affine3d grasp_translation, grasp_translation1, grasp_translation2, grasp_translation3;
        geometry_msgs::Pose grasp_pose;
        geometry_msgs::PoseArray pose_array;
        pose_array.header.frame_id = "/world";
        pose_array.header.stamp = ros::Time::now();

        std::vector<geometry_msgs::Pose> waypoints1, waypoints2;

        grasp_translation = Eigen::Translation3d(Eigen::Vector3d(0.0,0.0,-capsicum_model[2]/2 -cutting_offset[2] -0.02));
        Eigen::Affine3d grasp_pose_eigen1 = grasp_pose_eigen*grasp_translation;
        tf::poseEigenToMsg(grasp_pose_eigen1,grasp_pose);

        waypoints1.push_back(grasp_pose);
        pose_array.poses.push_back(grasp_pose);

        grasp_translation1 = Eigen::Translation3d(Eigen::Vector3d(0.0,0.0,-capsicum_model[2]/2 -cutting_offset[2]));
        Eigen::Affine3d grasp_pose_eigen2 = grasp_pose_eigen*grasp_translation1;
        tf::poseEigenToMsg(grasp_pose_eigen2,grasp_pose);

        waypoints1.push_back(grasp_pose);
        pose_array.poses.push_back(grasp_pose);

        vis_pub.publish(pose_array);
        pose_array.header.stamp = ros::Time::now();


        if(!moveToCartesianPath(waypoints1,velocity_cut, attempts, wait_time, false, 0)){return false;}

        if(!turnOnIO(CUTTER_PIN));

        grasp_translation2 = Eigen::Translation3d(Eigen::Vector3d(cutting_offset[0],0.0,-capsicum_model[2]/2 -cutting_offset[2]));
        Eigen::Affine3d grasp_pose_eigen3 = grasp_pose_eigen*grasp_translation2;
        tf::poseEigenToMsg(grasp_pose_eigen3,grasp_pose);
        waypoints2.push_back(grasp_pose);
        pose_array.poses.push_back(grasp_pose);


        grasp_translation3 = Eigen::Translation3d(Eigen::Vector3d(-cutting_offset[0],0.0,-capsicum_model[2]/2 -cutting_offset[2]));
        Eigen::Affine3d grasp_pose_eigen4 = grasp_pose_eigen*grasp_translation3;
        tf::poseEigenToMsg(grasp_pose_eigen4,grasp_pose);
        waypoints2.push_back(grasp_pose);
        pose_array.poses.push_back(grasp_pose);

        vis_pub.publish(pose_array);
        if(!moveToCartesianPath(waypoints2,velocity_cut, attempts, wait_time, false, 0)){return false;}

         if(!turnOffIO(CUTTER_PIN));
        return true;

    }


    void publishPosefromEigen(Eigen::Affine3d pose_eigen, std::string frame_id)
    {
        geometry_msgs::PoseArray pose_array;
        geometry_msgs::Pose pose;

        tf::poseEigenToMsg(pose_eigen,pose);
        pose_array.header.frame_id = frame_id;
        pose_array.header.stamp = ros::Time::now();
        pose_array.poses.push_back(pose);
        vis_pub.publish(pose_array);
    }

    bool segmentCapsicum(sensor_msgs::PointCloud2 msg_in, sensor_msgs::PointCloud2 &msg_out){
        bool ret = false;
        capsicum_detection::segmentCapsicum srv;
        srv.request.cloud = msg_in;
        if(segmentCapsicum_client.call(srv))
        {
            ret = true;
            ROS_INFO("Got Segmented Capsicum Response");
            msg_out = srv.response.segmented_cloud;

        }else{
            ROS_ERROR("Did not receive a response from the server");
            ret = false;
        }
        return ret;
    }

    bool estimateCapsicum(sensor_msgs::PointCloud2 cloud, Eigen::Affine3d &capsicum_pose_eigen, Eigen::Vector3d &capsicum_model)
    {
        bool ret = false;
        superquadric_fitter::estimateCapsicumPose srv;
        srv.request.cloud = cloud;
        if(estimateCapsicumPose_client.call(srv))
        {
            ret = true;
            ROS_INFO("Got Capsicum Pose Response");

            tf::transformMsgToEigen(srv.response.transform,capsicum_pose_eigen);

            capsicum_model << srv.response.a, srv.response.b, srv.response.c;
        }else{
            ROS_ERROR("Did not receive a response from the server");
            ret = false;
        }
        return ret;
    }

    bool scanCapsicum(Eigen::Vector4d start_q_eigen, Eigen::Vector3d delta_xyz, Eigen::Vector3d scan_offset, geometry_msgs::PointStamped target_centroid){

        Eigen::Vector3d scan_xyz;

        geometry_msgs::Quaternion start_q, q_left,q_right,q_up,q_down;
        std::vector<geometry_msgs::Pose> waypoints;

        geometry_msgs::PoseArray pose_array;
        geometry_msgs::Pose scan_pose;
        pose_array.header.frame_id = "/world";
        pose_array.header.stamp = ros::Time::now();


        double angle = 8*M_PI/180;
        //angle = atan(delta_xyz[1]/scan_offset[0])/2;

        start_q.x = start_q_eigen[0]; start_q.y = start_q_eigen[1]; start_q.z = start_q_eigen[2]; start_q.w = start_q_eigen[3];

        q_left.x = cos(-angle); q_left.y = sin(-angle); q_left.z = 0.0; q_left.w = 0.0;
        q_up.x = cos(-angle); q_up.y = 0.0; q_up.z = sin(-angle); q_up.w = 0.0;
        q_right.x = cos(angle); q_right.y = sin(angle); q_right.z = 0.0; q_right.w = 0.0;
        q_down.x = cos(angle); q_down.y = 0.0; q_down.z = sin(angle); q_down.w = 0.0;

        scan_pose.position.x += target_centroid.point.x - scan_offset[0];
        scan_pose.position.y += target_centroid.point.y - scan_offset[1];
        scan_pose.position.z += target_centroid.point.z - scan_offset[2];

        scan_pose.orientation = start_q;



        ROS_INFO("Starting to scan capsicum.");

        //Build up multiple views of capsicum
         ur5.setMaxVelocityScalingFactor(velocity_scan);

         //Start waypoint
         waypoints.push_back(scan_pose);
         pose_array.poses.push_back(scan_pose);

         //Waypoint scan left
         scan_pose.position.y += delta_xyz[1];
         scan_pose.orientation = q_left;
         waypoints.push_back(scan_pose);
         pose_array.poses.push_back(scan_pose);

         //Waypoint scan up
         scan_pose.position.y -= delta_xyz[1];
         scan_pose.position.z += delta_xyz[2];
         scan_pose.orientation = q_up;
         waypoints.push_back(scan_pose);
         pose_array.poses.push_back(scan_pose);

         //Waypoint scan right
         scan_pose.position.z -= delta_xyz[2];
         scan_pose.position.y -= delta_xyz[1];
         scan_pose.orientation = q_right;
         waypoints.push_back(scan_pose);
         pose_array.poses.push_back(scan_pose);

         // scan down
         scan_pose.position.y += delta_xyz[1];
         scan_pose.position.z -= delta_xyz[2]*3.0/4.0;
         scan_pose.orientation = q_down;
         waypoints.push_back(scan_pose);
         pose_array.poses.push_back(scan_pose);


         //Waypoint back to start;
         scan_pose.position.z += delta_xyz[2]*3.0/4.0;
         scan_pose.orientation = start_q;
         waypoints.push_back(scan_pose);
         pose_array.poses.push_back(scan_pose);

         vis_pub.publish(pose_array);

         if(!moveToCartesianPath(waypoints,velocity_scan, 20, wait_time, false, 0)){return false;}

         ROS_INFO("Finished scanning capsicum.");

         return true;

    }


  bool moveToCartesianPath(std::vector<geometry_msgs::Pose> waypoints, double velocity_scale, int Attempts, float timeout, bool async, int digital_pin){

      moveit_msgs::RobotTrajectory trajectory_msg;       // eef_step // jump_threshold
      double fraction_complete;
      int i = 0;


//      while(i < Attempts){

        fraction_complete = ur5.computeCartesianPath(waypoints, eef_step_size,  jump_threshold, trajectory_msg);

//        if(fraction_complete < 1.0) {
//            i++;
//            //eef_step_size += 0.01;
//            //jump_threshold += 0.5;
//            ROS_INFO_STREAM("Failed to plan cartesian path: Percent Complete:"
//                            << fraction_complete*100.0
//                            << " Increasing tresholds: eef_step:" << eef_step_size
//                            << " Jump threshold: " << jump_threshold);
//        }else{
//            break;
//        }
//      }

      // The trajectory needs to be modified so it will include velocities as well.
      // First to create a RobotTrajectory object
      robot_trajectory::RobotTrajectory rt(ur5.getCurrentState()->getRobotModel(), "manipulator");

      // Second get a RobotTrajectory from trajectory
      rt.setRobotTrajectoryMsg(*ur5.getCurrentState(), trajectory_msg);

      // Thrid create a IterativeParabolicTimeParameterization object
      trajectory_processing::IterativeParabolicTimeParameterization iptp;

      // Fourth compute computeTimeStamps
      bool success = iptp.computeTimeStamps(rt,velocity_scale);
      ROS_INFO("Computed time stamp %s",success?"SUCCEDED":"FAILED");

      // Get RobotTrajectory_msg from RobotTrajectory
      rt.getRobotTrajectoryMsg(trajectory_msg);

      // Finally plan and execute the trajectory
      ur5_plan.trajectory_ = trajectory_msg;
      ROS_INFO("Visualizing plan 4 (cartesian path) (%.2f%% acheived)",fraction_complete * 100.0);

      try{
#ifdef DEBUG
          cout << "Press the ENTER key to move arm or r to RESET";
          unsigned char c;
          c << cin.get();
//          while( c << cin.get()  ) {
//            if(c == '\n') break;
//          }
#endif

          ROS_INFO("Moving");
          if(async){

            ros::Duration trajectory_time = trajectory_msg.joint_trajectory.points.back().time_from_start;
            ros::Duration duration_offset(2.0);
            ROS_INFO("Moving asyncronously");
            if(!asyncExecuteUntilDigitalIO(digital_pin, trajectory_time + duration_offset, waypoints.back())) return false;

          }else{
            ROS_INFO("Moving without async");
            ur5.execute(ur5_plan);
          }

          return true;
      }
      catch(moveit::planning_interface::MoveItErrorCode ex){
          std::cout << "Something went wrong. Failed to move to pose" << std::endl;
          return false;
      }
  }

bool waitForDigitalPin(int digital_pin, ros::Duration timeout, geometry_msgs::Pose goal_pose){

    ros::Time start_time = ros::Time::now();
    ros::Rate rate(20);
    bool ret = false;

    while(!digital_in_states[digital_pin].state ){

        if(comparePose(goal_pose,ur5.getCurrentPose().pose, 0.01)){
            ret = false;
            break;
        }

        if(ros::Time::now() - start_time > timeout){
            ret = false;
            break;
        }
        rate.sleep();
    }

    return ret;
}

bool comparePose(geometry_msgs::Pose pose1, geometry_msgs::Pose pose2, double tolerance){
    double distance = sqrt(pow(pose1.position.x -pose2.position.x,2)
                         + pow(pose1.position.y -pose2.position.y,2)
                         + pow(pose1.position.z -pose2.position.z,2));

    if(distance < tolerance){
        return true;
    }else{
        return false;
    }
}

  bool asyncExecuteUntilDigitalIO(int digital_pin, ros::Duration timeout, geometry_msgs::Pose goal_pose){

      try{

#ifdef DEBUG
          cout << "Press the ENTER key to move arm or r to RESET";

          unsigned char c;
          while( c << cin.get()  ) {
            if(c == '\n') break;
            else if(c == 'r'); updateState(RESET); return false;
          }
#endif
        ur5.asyncExecute(ur5_plan);
        if(waitForDigitalPin(digital_pin,timeout,goal_pose)){
            return true;
        }else{
            return false;
        }
      }
      catch(moveit::planning_interface::MoveItErrorCode ex){
          std::cout << "Something went wrong. Failed to move to pose" << std::endl;
          return false;
      }

  }

  //moves arm while constraining the end effector orientation to the qConstraint variable
  bool moveToOrientationConstraint(geometry_msgs::PoseStamped pose, Eigen::Vector4d qConstraint, int Attempts, float timeout){

      moveit_msgs::OrientationConstraint ocm;
      ocm.link_name = "ee_link";
      ocm.header.frame_id = "world";
      ocm.orientation.x = qConstraint[0];
      ocm.orientation.y = qConstraint[1];
      ocm.orientation.z = qConstraint[2];
      ocm.orientation.w = qConstraint[3];
      ocm.absolute_x_axis_tolerance = 0.5;
      ocm.absolute_y_axis_tolerance = 0.5;
      ocm.absolute_z_axis_tolerance = 0.5;
      ocm.weight = 1.0;

      moveit_msgs::Constraints constraints;
      constraints.orientation_constraints.push_back(ocm);
      ur5.setPathConstraints(constraints);

      bool ret = false;
      bool success = false;

      while(Attempts > 0){
          ur5.setStartStateToCurrentState();
          pose.header.stamp = ros::Time::now();
          ur5.setPoseTarget(pose);

          success = ur5.plan(ur5_plan);

          if(success){
              ROS_INFO("Successfully planned to goal");
              if(executeMove()) {
                  Attempts = 0;
                  ret = true;
              }else{
                  return false;
              }
          }else{
              Attempts--;
              sleep(timeout);
          }
      }
      return ret;

      ur5.clearPathConstraints();

  }



  bool moveToNamed(std::string namedGoal, int Attempts, float timeout){

	bool ret = false;

	while(Attempts > 0){
        ur5.setStartStateToCurrentState();
        ur5.setNamedTarget(namedGoal);

        if(ur5.plan(ur5_plan)){
            std::cout << "Successfully planned to " << namedGoal << std::endl;
            ROS_INFO("Successfully planned to goal");
            if(executeMove()) {
                Attempts = 0;
                ret = true;
            }else{
                return false;
            }
        }else{
            std::cout << "Failed to plan, trying to replan" << std::endl;
            Attempts--;
            sleep(timeout);
        }

	}
	return ret;

  }

  bool moveTo(Eigen::Affine3d pose_eigen, std::string frame_id, int Attempts, double timeout)
  {
    geometry_msgs::PoseStamped poseStamped;
    poseStamped.header.stamp = ros::Time::now();
    poseStamped.header.frame_id = frame_id;
    tf::poseEigenToMsg(pose_eigen,poseStamped.pose);

    return moveTo(poseStamped,Attempts,timeout);
  }

  bool moveTo(geometry_msgs::PoseStamped pose, int Attempts, float timeout)
  {
	bool ret = false;
    bool success = false;

	while(Attempts > 0){
        ur5.setStartStateToCurrentState();
        pose.header.stamp = ros::Time::now();
        ur5.setPoseTarget(pose);

        success = ur5.plan(ur5_plan);

		if(success){
            ROS_INFO("Successfully planned to goal");
            if(executeMove()) {
                Attempts = 0;
                ret = true;
            }else{
                return false;
            }
		}else{
			Attempts--;
            sleep(timeout);
		}
	}
	return ret;
  }

  bool moveTo(std::string frame_id, Eigen::Vector3d position, Eigen::Vector4d orientation, int Attempts, double timeout){

    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = frame_id;
    pose.pose.position.x = position[0];
    pose.pose.position.y = position[1];
    pose.pose.position.z = position[2];

    pose.pose.orientation.x = orientation[0];
    pose.pose.orientation.y = orientation[1];
    pose.pose.orientation.z = orientation[2];
    pose.pose.orientation.w = orientation[3];

    return moveTo(pose, Attempts, timeout);

  }

  bool executeMove(){

      try{
#ifdef DEBUG
          cout << "Press the ENTER key to move arm or r to RESET";
          unsigned char c;
          while( c << cin.get()  ) {
            if(c == '\n') break;
            else if(c == 'r'); updateState(RESET); return false;
          }
#endif
          ur5.move();
          std::cout << "Moved to Goal" << std::endl;
          return true;
      }
      catch(moveit::planning_interface::MoveItErrorCode ex){
          std::cout << "Something went wrong. Failed to move to pose" << std::endl;
          return false;
      }

  }


  void publishTFfromPose(Eigen::Affine3d pose, std::string parent_frame, std::string child_frame)
  {
      static tf::TransformBroadcaster tf_broadcaster;

      tf::Transform transform;
      tf::transformEigenToTF(pose,transform);

      tf_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), parent_frame, child_frame));

  }

  void createGraspTransform(Eigen::Affine3d &capsicum_transform, Eigen::Vector3d &capsicum_model, Eigen::Affine3d &grasp_transform, Eigen::Vector3d &grasp_offset){

    Eigen::Matrix3d capsicum_rotation = capsicum_transform.rotation();
    Eigen::Vector3d capsicum_swap_model = capsicum_model;
    Eigen::Matrix3d grasp_pose_rotation;
    Eigen::Vector3d grasp_pose_translation;

    int frontAxis, sideAxis, stemAxis;
    capsicum_rotation.row(0).cwiseAbs().maxCoeff(&frontAxis);
    capsicum_rotation.row(1).cwiseAbs().maxCoeff(&sideAxis);
    capsicum_rotation.row(2).cwiseAbs().maxCoeff(&stemAxis);

    Eigen::Vector3d frontVector = capsicum_rotation.col(frontAxis);
    Eigen::Vector3d stemVector = capsicum_rotation.col(stemAxis);
    Eigen::Vector3d sidevector = capsicum_rotation.col(sideAxis);

    double angleFrontAxis = acos(frontVector[0]/frontVector.norm());
    double angleSideAxis = acos(sidevector[1]/sidevector.norm());
    double angleStemAxis = acos(stemVector[2]/stemVector.norm());

    //Keep track of geometry model using front side and stem axes
    capsicum_model[0] = capsicum_swap_model[frontAxis];
    capsicum_model[1] = capsicum_swap_model[sideAxis];
    capsicum_model[2] = capsicum_swap_model[stemAxis];


    ROS_INFO_STREAM("Capsicum Rotation Matrix: "<< capsicum_rotation);
    ROS_INFO_STREAM("Grasp Pose Rotation Matrix: "<< grasp_pose_rotation);


    std::vector<std::string> Axis = {"x","y","z"};
    ROS_INFO_STREAM("Angle to front face: " << angleFrontAxis*180/M_PI << " Capsicum front axis: " << Axis[frontAxis]);
    ROS_INFO_STREAM("Angle to side face: " << angleSideAxis*180/M_PI << " Capsicum side axis: " << Axis[sideAxis]);
    ROS_INFO_STREAM("Angle to stem face: " << angleStemAxis*180/M_PI << " Capsicum stem axis: " << Axis[stemAxis]);

    if(angleFrontAxis < M_PI/2) grasp_pose_rotation.col(0) = capsicum_rotation.col(frontAxis);
    else grasp_pose_rotation.col(0) = -1*capsicum_rotation.col(frontAxis);

    if(angleStemAxis > M_PI/2) grasp_pose_rotation.col(2) = capsicum_rotation.col(stemAxis);
    else grasp_pose_rotation.col(2) = -1*capsicum_rotation.col(stemAxis);

    grasp_pose_rotation.col(1) = -1*(grasp_pose_rotation.col(0).cross(grasp_pose_rotation.col(2)));

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
    transform.block<3,3>(0,0) = grasp_pose_rotation; //grasp_pose_rotation;
    transform.block<3,1>(0,3) = capsicum_transform.translation();
    Eigen::Affine3d grasp_transform_eigen(transform);

    //grasp_offset *= -1;
    //grasp_offset[frontAxis] = -1*grasp_offset[frontAxis] - model_abc[frontAxis];
    translation.block<3,1>(0,3) = -1*grasp_offset ;
    Eigen::Affine3d transform_translation(translation);

    publishTFfromPose(grasp_transform_eigen*transform_translation, "/world", "/gripper_frame");

    grasp_transform = grasp_transform_eigen*transform_translation;

    //tf::poseEigenToMsg(grasp_transform*transform2,grasp_pose);

//    ROS_INFO_STREAM("Capsicum Rotation Matrix: "<< capsicum_rotation);
//    ROS_INFO_STREAM("Grasp Pose Rotation Matrix: "<< grasp_pose_rotation);

  }


  void attachCapsicumObject(){
      ur5.attachObject("capsicum","ee_link");
  }

  void dettachCapsicumObject(){
      ur5.detachObject("capsicum");
  }

  void removeCapsicumObject(){
      std::vector<std::string> object_names = {"capsicum"};
      planning_scene_interface.removeCollisionObjects(object_names);
  }

  void addBoxObject(){

      moveit_msgs::CollisionObject collision_object;
      collision_object.header.frame_id = ur5.getPlanningFrame();
      collision_object.id = "box";

      shape_msgs::SolidPrimitive primitive;
      primitive.type = primitive.BOX;
      primitive.dimensions.resize(3);

      primitive.dimensions[0] = 0.45;
      primitive.dimensions[1] = 0.9;
      primitive.dimensions[2] = 0.55;

      geometry_msgs::Pose object_pose;


      object_pose.position.x = 0.95;
      object_pose.position.y = 0;
      object_pose.position.z = -0.575;

      collision_object.primitives.push_back(primitive);
      collision_object.primitive_poses.push_back(object_pose);
      collision_object.operation = collision_object.ADD;

      collision_objects.push_back(collision_object);

      ROS_INFO("Add an object into the world");
      planning_scene_interface.addCollisionObjects(collision_objects);

  }

  void addCapsicumObject(geometry_msgs::Pose object_pose){

      moveit_msgs::CollisionObject collision_object;
      collision_object.header.frame_id = "/world";
      collision_object.id = "capsicum";

      shape_msgs::SolidPrimitive primitive;
      primitive.type = primitive.CYLINDER;
      primitive.dimensions.resize(2);
      primitive.dimensions[0] = 0.08;
      primitive.dimensions[1] = 0.04;

      object_pose.orientation.x = 3.14;
      object_pose.orientation.y = 0;
      object_pose.orientation.z = 0;


      collision_object.primitives.push_back(primitive);
      collision_object.primitive_poses.push_back(object_pose);
      collision_object.operation = collision_object.ADD;

      collision_objects.push_back(collision_object);

      ROS_INFO("Add an object into the world");
      planning_scene_interface.addCollisionObjects(collision_objects);

  }


  void openWireCutter(int8_t pin){
      std_msgs::Float64 servo_angle;
      servo_angle.data = 5.2;
      wire_cutter_pub.publish(servo_angle);
  }


  void closeWireCutter(int8_t pin){
      std_msgs::Float64 servo_angle;
      servo_angle.data = 0.0;
      wire_cutter_pub.publish(servo_angle);
  }


 bool turnOnIO(int8_t pin){

     ur_msgs::SetIO io_srv;
     bool success = false;

     io_srv.request.fun = 1;
     io_srv.request.pin = pin;
     io_srv.request.state = true;


     if(io_client.call(io_srv)){
         success = io_srv.response.success;
     }

     return success;
 }

 bool turnOffIO(int8_t pin){

     ur_msgs::SetIO io_srv;
     bool success = false;

     io_srv.request.fun = 1;
     io_srv.request.pin = pin;
     io_srv.request.state = false;


     if(io_client.call(io_srv)){
         success = io_srv.response.success;
     }

     return success;
 }

 void io_states_callback(const ur_msgs::IOStates msg){
     digital_in_states = msg.digital_in_states;
     ROS_INFO_STREAM("Pressure Switch" << digital_in_states[PRESSURE_SWITCH].state);
 }


 void scene_cloud_callback(const sensor_msgs::PointCloud2ConstPtr msg){
    scene_cloud_msg = *msg;
 }

 bool moveToInitCapsicum(geometry_msgs::PointStamped target_capsicum_centroid){

    geometry_msgs::PoseStamped pose;
    geometry_msgs::PoseArray pose_array;
     pose_array.header.frame_id = "/world";
     pose_array.header.stamp = ros::Time::now();

     pose.header.frame_id = "/world";
     pose.header.stamp = ros::Time::now();

     pose.pose.position.x += target_capsicum_centroid.point.x - scan_offset[0];
     pose.pose.position.y += target_capsicum_centroid.point.y - scan_offset[1];
     pose.pose.position.z += target_capsicum_centroid.point.z - scan_offset[2];

     pose.pose.orientation.x = 1;
     pose.pose.orientation.y = 0;
     pose.pose.orientation.z = 0;
     pose.pose.orientation.w = 0;

     //std::vector<geometry_msgs::Pose> waypoints;

     //waypoints.push_back(pose);
     pose_array.poses.push_back(pose.pose);
     vis_pub.publish(pose_array);
     pose_array.header.stamp = ros::Time::now();

     ur5.setMaxVelocityScalingFactor(velocity_scan);
     if(!moveTo(pose, attempts, wait_time)){return false;}
     //moveToCartesianPath(waypoints,velocity_scan, attempts, wait_time, false, 0)){return false;}

 }


 bool detect_init_capsicum_from_image(cv::Mat scene_image, float focal_x, float focal_y, float depth, geometry_msgs::PointStamped &capsicum_point){

     cv::Mat highlighted = capsicumDetector.detect(scene_image,200);
     geometry_msgs::PointStamped capsicum_point_camera;
     geometry_msgs::Pose capsicum_pose;
     geometry_msgs::PoseArray pose_array;
     pose_array.header.frame_id = "/world";
     pose_array.header.stamp = ros::Time::now();

     int selected_capsicum = 0;
     int rows = scene_image.rows;
     int cols = scene_image.cols;
     bool found_capsicum = false;

     capsicum_pose.orientation.x = 1;

     if(capsicumDetector.nCapsicumFound >= 1){
         found_fruit = true;
         capsicums.clear();

         for( int j = 0; j < capsicumDetector.nCapsicumFound; j++ ) {
             capsicum cap(capsicumDetector.mu[j], capsicumDetector.boundRect[j]);
             capsicums.push_back(cap);
         }
     }else{
        ROS_INFO("Can not find any capsicums in current image");
	return false;
     }

     double closest_distance = 1e6;

     //selecting best capsicum to harvest based on area and location
     for( int j = 0; j < capsicumDetector.nCapsicumFound; j++ ) {

         double distance = sqrt(pow(capsicums[j].center_point.x -cols/2,2) + pow(capsicums[j].center_point.y -rows/2,2));

         if((capsicums[j].area > 5000) && (distance < closest_distance)){
                 ROS_INFO_STREAM("Found Capsicum with distance:" << distance
                                 << "index: " << j
                                 << "area: " << capsicums[j].area);
                 closest_distance = distance;
                 selected_capsicum = j;
                 found_capsicum = true;
         }

     }

//     for (std::vector<PointCloud::Ptr>::iterator it = clusters.begin (); it != clusters.end (); ++it){
//         PointCloud::Ptr cloud = *it;
//         Eigen::Vector4f centroid;
//         pcl::compute3DCentroid(*cloud, centroid);
//         euclidean_distance = sqrt(pow(centroid[0]-0.25,2) + pow(centroid[1]-0.25,2) + pow(centroid[2],2));
//         if((euclidean_distance < closest_capsicum) && (cloud->size() >= 0.5*mean_size)){
//           closest_capsicum = euclidean_distance;
//           cout << "Closest capsicum is " << closest_capsicum << endl;
//           capsicum_cloud = cloud;
//         }
//         //if(cloud->size() > capsicum_cloud->size()) capsicum_cloud = cloud;

//     }
     if(!found_capsicum){
         return false;
     }

     int image_x = capsicums[selected_capsicum].center_point.x;
     int image_y = capsicums[selected_capsicum].center_point.y;


     //Using planar assumption at depth the relationship is (focal_x/depth = ximage/xcartesian)
     capsicum_point_camera.point.y = -(image_x - cols/2)*depth/focal_x;
     capsicum_point_camera.point.z = -(image_y - rows/2)*depth/focal_y;
     capsicum_point_camera.point.x = depth;
     capsicum_point_camera.header.frame_id = "/camera_link";

//     ROS_INFO_STREAM("Found Capsicums at xyz camera:" << capsicum_point_camera.point);

     capsicum_point.header.frame_id = "/world";
     tf_listener.transformPoint("/world",capsicum_point_camera,capsicum_point);

     capsicum_pose.position = capsicum_point.point;
     pose_array.poses.push_back(capsicum_pose);
     vis_pub.publish(pose_array);

//     ROS_INFO_STREAM("Converted point to world frame at xyz:" << capsicum_point.point);

     //cv::imshow("capsicums", highlighted);
     //cv::waitKey(10);

     return true;

 }

 bool detect_init_capsicum(geometry_msgs::PointStamped &capsicum_centroid){

     sensor_msgs::PointCloud2 segmented_capsicum;

     if(!segmentCapsicum(scene_cloud_msg,segmented_capsicum)){
        return false;
     }

     if(segmented_capsicum.height*segmented_capsicum.width > 0)
     {
        PointCloud::Ptr segmented_capsicum_pcl(new PointCloud);
        Eigen::Vector4f centroid;
        pcl::fromROSMsg(segmented_capsicum,*segmented_capsicum_pcl);
        pcl::compute3DCentroid(*segmented_capsicum_pcl, centroid);


        try{

            geometry_msgs::PointStamped centroid_point;
            centroid_point.header.frame_id = "/kf_world";
            centroid_point.point.x = centroid[0];
            centroid_point.point.y = centroid[1];
            centroid_point.point.z = centroid[2];

            //target_capsicum_centroid.header.frame_id = "world";
            tf_listener.transformPoint("/world",centroid_point,capsicum_centroid);
            return true;

        }catch(tf::TransformException ex){
          ROS_ERROR("%s",ex.what());
          return false;
        }

     }else{
         return false;
     }


 }

 void find_fruit_callback(const sensor_msgs::PointCloud2ConstPtr msg){


     while(!found_fruit && (msg->height*msg->width > 0))
     {
        PointCloud::Ptr cloud(new PointCloud);
        PointCloud::Ptr capsicum_cloud(new PointCloud);
        Eigen::Vector4f centroid;
        pcl::fromROSMsg(*msg,*capsicum_cloud);
        pcl::compute3DCentroid(*capsicum_cloud, centroid);


        try{

            geometry_msgs::PointStamped centroid_point, world_point;
            centroid_point.header.frame_id = "/kf_world";
            centroid_point.point.x = centroid[0];
            centroid_point.point.y = centroid[1];
            centroid_point.point.z = centroid[2];

            //target_capsicum_centroid.header.frame_id = "world";
            tf_listener.transformPoint("world",centroid_point,target_capsicum_centroid);
            found_fruit = true;

        }catch(tf::TransformException ex){
          ROS_ERROR("%s",ex.what());
        }
        //pcl::PointXYZ centre_point(centroid[0],centroid[1],centroid[2]);

     }


 }

 void image_callback(const sensor_msgs::ImageConstPtr& msg){

     readImage(msg,scene_image);

 }

 void image_callback_detect(const sensor_msgs::ImageConstPtr& msg){

     cv::Mat rgb_image_;
     readImage(msg,rgb_image_);

     cv::Mat highlighted = capsicumDetector.detect(rgb_image_,200);

     if(capsicumDetector.nCapsicumFound >= 1){
         found_fruit = true;
         capsicums.clear();

         for( int j = 0; j < capsicumDetector.nCapsicumFound; j++ ) {
             capsicum cap(capsicumDetector.mu[j], capsicumDetector.boundRect[j]);
             capsicums.push_back(cap);
             //capsicums.insert(it,1,cap);
             //it++;
         }
     }


     cv::imshow(OPENCV_WINDOW, highlighted);
     cv::waitKey(1);

 }


  void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image)
  {
      cv_bridge::CvImageConstPtr pCvImage;
      pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
      pCvImage->image.copyTo(image);
  }


  void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix)
  {
      double *itC = cameraMatrix.ptr<double>(0, 0);
      for(size_t i = 0; i < 9; ++i, ++itC)
      {
        *itC = cameraInfo->K[i];
      }
  }

  void mouse_click(int event, int x, int y)
  {

      switch(event)
      {
          case CV_EVENT_LBUTTONDOWN:
          {
              std::cout<<"Mouse Pressed at x:"<< x << std::endl;
              std::cout<<"Mouse Pressed at y:"<< y << std::endl;
              if(found_fruit){
                  for (int i=0; i<capsicumDetector.nCapsicumFound; i++){
                      if((capsicums[i].center_point.x < x + 50)&&(capsicums[i].center_point.x > x-50)&&
                              (capsicums[i].center_point.y < y + 50)&&(capsicums[i].center_point.y > y-50)){
                                std::cout<<"You have selected to pick fruit at x,y:"<< x << ", " << y << std::endl;
                                pick_fruit = true;
                                pick_fruit_index = i;

                      }
                  }
              }

              break;
          }
      }

  }

  static void mouse_click(int event, int x, int y, int, void* this_) {
    static_cast<harvey*>(this_)->mouse_click(event, x, y);
  }

};



int main(int argc, char** argv)
{
  std::vector<capsicum> caps; //= new std::vector<capsicum>();

  ros::init(argc, argv, "field_trial_node");
  harvey fp;

  fp.start();

  return 0;
}
