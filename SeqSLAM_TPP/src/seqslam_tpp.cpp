/*
Copyright 2016 Australian Centre for Robotic Vision
Author: James Sergeant james.sergeant@qut.edu.au
*/

// REMEMBER to set ROS_MASTER_URI and ROS_IP, also source
// harvey_ws/src/.external on UR5
// roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch
// limited:=true
// roslaunch ur_description ur5_upload.launch limited:=true
// roslaunch ur5_moveit_config moveit_rviz.launch config:=true limited:=true
// rosrun seqslam_tpp seqslam_tpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <moveit/move_group_interface/move_group.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "tf/tf.h"

#include <seqslam_tpp/UserSelection.h>
#include <seqslam_tpp/UserSelectionRequest.h>
#include <seqslam_tpp/UserSelectionResponse.h>
#include <seqslam_tpp/MATLABSrv.h>
#include <seqslam_tpp/MATLABSrvRequest.h>
#include <seqslam_tpp/MATLABSrvResponse.h>

#define _USE_MATH_DEFINES

#define HALF_HFOV 27  // half horizontal FOV of the camera
#define HALF_VFOV 18  // half vertical FOV of the camera
#define CAMERA_OFFSET \
    0.10  // additional offset for camera, not included in model
// #define IM_WIDTH 400          // width of image used for SeqSLAM
// registration

static const std::string HI_RES_WINDOW = "High Resolution Camera";
static const std::string LO_RES_WINDOW =
    "Ref Pos: Right-Click, Sensor Pos: Middle-Click";
static const std::string REGION_WINDOW = "Selected Region Of Interest";
static const std::string FEATURE_WINDOW = "Feature Region";

class SeqSLAM_TPP {
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    tf::TransformListener tf_listener;

    ros::Subscriber imageHRSub;
    ros::Subscriber imageLRSub;
    ros::Subscriber matlabSub;
    ros::Subscriber servoSub;

    image_transport::Publisher initialImagePub;
    image_transport::Publisher secondaryImagePub;
    image_transport::Publisher servoPub;

    ros::Publisher scalePub;
    ros::Publisher recordPub;

    ros::ServiceServer userInputRequestSrv;
    ros::ServiceServer getImageSrv;
    ros::ServiceServer matlabSrv;

    sensor_msgs::ImagePtr watsonMsg;
    sensor_msgs::ImagePtr pixlMsg;
    sensor_msgs::ImagePtr servoMsg;
    sensor_msgs::Image points;
    sensor_msgs::Image imgMsg;

    std_msgs::Float32MultiArray scale;
    std_msgs::String recordMsg;

    cv::Point pointA;
    cv::Point pointB;
    cv::Point featurePoint;

    std::ostringstream note;

    std::ofstream result_file;

    std::string menuSelect;
    std::string operator_success;
    std::string image_topic;
    std::string robot_name;
    std::string move_group_name;

    bool reset_pixl;
    bool reset_watson;
    bool reg_complete;
    bool reg_success;
    bool servo_success;
    bool servoing;
    bool initialise;
    bool regionSelected= false;

    std::vector<moveit_msgs::CollisionObject> collision_objects;
    std::vector<geometry_msgs::Pose> waypoints;
    std::vector<float> servo_data;

    float Froi;
    float initialImageScale;
    float watsonScaleFromRef;
    float watsonScaleFromROI;
    float secondaryImageScale;
    float secondImageMult;
    float initialImageMult;
    float objectposz;
    float objectposzInit;
    float wait_time = 1.0;
    float servo_scale;
    float servo_rotation;
    float servo_x;
    float servo_y;

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

    int servo_attempts;
    int time_id;

    time_t timer;

    XmlRpc::XmlRpcValue poses;

    // // servo thresholds - tight
    // float thresh[4] = {0.01,1,5,5};

    // servo thresholds - LOOSE
    float thresh[4] = {0.03, 3, 15, 15};

   public:
    SeqSLAM_TPP() : it_(nh_) {
        cv::namedWindow(HI_RES_WINDOW, cv::WINDOW_NORMAL);
        cv::namedWindow(LO_RES_WINDOW, cv::WINDOW_NORMAL);

        // obtain parameters
        nh_.param("/seqslam_tpp/image_topic", image_topic,
                  std::string("/usb_cam/image_raw"));
        nh_.param("/seqslam_tpp/robot", robot_name, std::string("baxter"));
        nh_.param("/seqslam_tpp/move_group", move_group_name,
                  std::string("right_arm"));
        nh_.param("/seqslam_tpp/" + robot_name, poses);

        // start Image Transports
        initialImagePub = it_.advertise("/seqslam_tpp/image_watson", 1);
        secondaryImagePub = it_.advertise("/seqslam_tpp/image_pixl", 1);
        servoPub = it_.advertise("/seqslam_tpp/image_servo", 1);

        // start Publishers
        scalePub =
            nh_.advertise<std_msgs::Float32MultiArray>("/seqslam_tpp/scale", 1);
        recordPub = nh_.advertise<std_msgs::String>("/seqslam_tpp/record", 1);

        imageHRSub =
            nh_.subscribe(image_topic, 10, &SeqSLAM_TPP::imageCallback, this);
        // servoSub = nh_.subscribe("/seqslam_tpp/servo_result", 10,
        //                          &SeqSLAM_TPP::servoCallback, this);

        // Start Services
        userInputRequestSrv = nh_.advertiseService(
            "/seqslam_tpp/user_input_request", &SeqSLAM_TPP::userInputRequestCallback, this);
        getImageSrv = nh_.advertiseService(
            "/seqslam_tpp/get_image", &SeqSLAM_TPP::getImageCallback, this);

        // change matlabSub to service call
        ros::ServiceClient matlabSub = nh_.serviceClient<seqslam_tpp::MATLABSrv>("/seqslam_tpp/matlab_result");
        // matlabSub = nh_.advertiseService("/seqslam_tpp/matlab_result", &SeqSLAM_TPP::matlabCallback, this);

        initialImageScale = 1.0;
        secondaryImageScale = 1.0;
        servoing = false;
        scale.data = {1.0,1.0};

        ros::Time time;

        note.str("");

        // object at 0 in world-z, gaussian noise
        // new_estimate();

        reset_pixl = false;
        initialise = true;

        ROS_INFO_STREAM("Object estimated at: " << objectposz << " meters.");

        ROS_INFO_STREAM("Init complete");

        ros::AsyncSpinner spinner(2);
        spinner.start();

        ros::waitForShutdown();
    }

    ~SeqSLAM_TPP() {
        cv::destroyWindow(HI_RES_WINDOW);
        cv::destroyWindow(LO_RES_WINDOW);
        cv::destroyWindow(REGION_WINDOW);
        // cv::destroyWindow(FEATURE_WINDOW);
    }

    void updateSendScale() {
        scale.data.clear();
        if (initialImageScale < secondaryImageScale) {
            secondImageMult = 1;
            initialImageMult = initialImageScale / secondaryImageScale;

        } else {
            secondImageMult = secondaryImageScale / initialImageScale;
            initialImageMult = 1;
        }
        scale.data.push_back(initialImageMult);
        scale.data.push_back(secondImageMult);

        scalePub.publish(scale);
    }

    bool userInputRequestCallback(seqslam_tpp::UserSelection::Request &req,
                            seqslam_tpp::UserSelection::Response &res) {
        cv::setMouseCallback(HI_RES_WINDOW, mouse_click, this);
        cv::setMouseCallback(LO_RES_WINDOW, pixl_click, this);

        while (!regionSelected) {

        }

        try {
            res.image = imgMsg;
            res.roi = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", regionOfInterest)
                                   .toImageMsg());

            res.roi_scale.data = Froi;

            res.bounding_box.top_left.x = pointA.x;
            res.bounding_box.top_left.y = pointA.y;
            res.bounding_box.bottom_right.x = pointB.x;
            res.bounding_box.bottom_right.y = pointB.y;

            res.message.data = "User selection returned";
            res.success.data = true;
        } catch (...) {
            ROS_WARN("Service Error");
            res.message.data = "Service Error";
            res.success.data = false;

        }

        regionSelected = false;

        // disable cv window callbacks
        cv::setMouseCallback(HI_RES_WINDOW, no_click, this);
        cv::setMouseCallback(LO_RES_WINDOW, no_click, this);

        return true;

    }

    bool getImageCallback(seqslam_tpp::UserSelection::Request &req,
                            seqslam_tpp::UserSelection::Response &res) {

        try {
            res.image = imgMsg;

            res.message.data = "Image returned";
            res.success.data = true;
        } catch (...) {
            ROS_WARN("Service Error");
            res.message.data = "Service Error";
            res.success.data = false;

        }

        return true;

    }

    void imageCallback(const sensor_msgs::Image::ConstPtr imageColor) {
        imgMsg = *imageColor;
        imageMsgToMat(imageColor, color);
        cv::imshow(HI_RES_WINDOW, color);
        cv::waitKey(3);
    }

    void matlabResponse(std_msgs::Float32MultiArray msg) {
        // set servo flag
        servoing = true;
        ROS_INFO_STREAM("SeqSLAM information received from MATLAB");

        // store scale, rot, Tx, Ty in usable form
        // ROS_INFO_STREAM(msg.data);
        std::vector<float> data;
        data = msg.data;
        Eigen::Map<Eigen::MatrixXf> mat(data.data(), msg.layout.dim[0].size,
                                        msg.layout.dim[1].size);
        // ROS_INFO_STREAM(mat);
        // if the initial registration fails, print message, cancel servoing,
        // make note
        if (mat(0, 0) == 0) {
            ROS_INFO_STREAM("Initial registration failed.");
            servoing = false;
            note << "Initial registration failed. ";
            return;
        }

        ROS_INFO_STREAM("MATLAB returned: " << mat);
        // move to estimated extension
        arm_pose = move_group.getCurrentPose();
        ROS_INFO_STREAM("secondaryImageScale: " << secondaryImageScale);
        // ROS_INFO_STREAM("obj estimate: " << objectposz);
        // objectposz = objectposz + (initialImageScale - initialImageScale*(1 +
        // (mat(0,0)-1)*initialImageMult/secondImageMult)) * (float(color.cols) /
        // regionOfInterest.cols);
        // ROS_INFO_STREAM("obj estimate updated: " << objectposz);
        ROS_INFO_STREAM("initialImageScale: " << initialImageScale / secondImageMult);
        initialImageScale /= mat(0, 0);  // (1 + (mat(0,0)-1)*initialImageMult);
        ROS_INFO_STREAM("initialImageScale updated: " << initialImageScale);
        // arm_pose.pose.position.z = - objectposz + initialImageScale +
        // CAMERA_OFFSET; // estimate vertical position based on returned scale
        arm_pose.pose.position.z =
            initialImageScale + CAMERA_OFFSET;  // estimate vertical position based on
        // returned scale
        ROS_INFO_STREAM("z: " << arm_pose.pose.position.z);
        float dx = (initialImageScale * tan(HALF_HFOV * M_PI / 180) * 2) *
                   (mat(0, 3) / color.cols);  // (estimate of width of image
        // (width) contents) *
        // (translation in pixels / image
        // width in pixels)
        float dy = (initialImageScale * tan(HALF_HFOV * M_PI / 180) * 2) *
                   (mat(0, 2) / color.cols);
        ROS_INFO_STREAM("dx: " << dx);
        ROS_INFO_STREAM("dy: " << dy);
        arm_pose.pose.position.x -= dx;
        arm_pose.pose.position.y += dy;  // sign due to world axes

        ROS_INFO_STREAM("Moving to estimated pose: \n" << arm_pose);

        try {
            if (!moveTo(arm_pose, 2, wait_time)) {
                throw tf::TransformException(
                    "Move to estimated position failed");
            }
        } catch (tf::TransformException ex) {
            // if move unsuccessful, reset to pixl then watson positions, record
            // to note,
            note << ex.what();
            ROS_INFO_STREAM(ex.what());
            servoing = false;
        }
        sleep(wait_time);  // avoid vibration in image
        if (!servoing) {
            ROS_INFO_STREAM("Not servoing");
        } else {
            est_pose = move_group.getCurrentPose();
            color.copyTo(pixl_est_image);
            // closed_servo(); // PID-controlled close loop servoing
            open_servo();  // OPEN LOOP "SERVOING"
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
    //
    // void closed_servo() {
    //     ROS_INFO_STREAM("Begin closed-loop servoing");
    //
    //     scale.data = {1.0,1.0};
    //     scalePub.publish(scale);
    //     servo_attempts = 10;
    //     servo_success = false;
    //     reg_success = false;
    //     std::vector<float> temp(4, 0);
    //     Eigen::ArrayXXf set_point(1, 4);
    //     set_point << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf err_sum(1, 4);
    //     err_sum << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf d_err(1, 4);
    //     d_err << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf lastErr(1, 4);
    //     lastErr << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf current(1, 4);
    //     current << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf error(1, 4);
    //     error << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf output(1, 4);
    //     output << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf kp(1, 4);
    //     kp << 0.05, 0.5, 0.00005, 0.00005;
    //     Eigen::ArrayXXf ki(1, 4);
    //     ki << 0.0, 0.0, 0.0, 0.0;
    //     Eigen::ArrayXXf kd(1, 4);
    //     kd << 0.0, 0.0, 0.00001, 0.00001;
    //
    //     while (!servo_success) {
    //         // servoMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8",
    //         // grey).toImageMsg();
    //         servoMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color)
    //                        .toImageMsg();
    //         servoPub.publish(servoMsg);
    //         reg_complete = false;
    //         while (!reg_complete) {
    //         }
    //
    //         if (reg_success) {
    //             arm_pose = move_group.getCurrentPose();
    //
    //             // Compute all the working error variables
    //             current << 1 - servo_scale, -servo_rotation, servo_x, servo_y;
    //
    //             error = set_point - current;
    //
    //             err_sum += error;
    //             d_err = (error - lastErr);
    //
    //             // /*Compute PID Output*/
    //             output = kp * error + ki * err_sum + kd * d_err;
    //
    //             tf::Quaternion q(
    //                 arm_pose.pose.orientation.x, arm_pose.pose.orientation.y,
    //                 arm_pose.pose.orientation.z, arm_pose.pose.orientation.w);
    //             tf::Matrix3x3 m(q);
    //             double roll, pitch, yaw;
    //             m.getRPY(roll, pitch, yaw);
    //
    //             arm_pose.pose.orientation =
    //                 tf::createQuaternionMsgFromRollPitchYaw(roll, pitch,
    //                                                         yaw + output(0, 1));
    //             arm_pose.pose.position.z -= output(0, 0);
    //             arm_pose.pose.position.x -= output(0, 2);
    //             arm_pose.pose.position.y -= output(0, 3);
    //
    //             ROS_INFO_STREAM("Moving to servo position");
    //
    //             try {
    //                 if (!moveTo(arm_pose, 2, wait_time)) {
    //                     throw tf::TransformException(
    //                         "Move to servo position failed");
    //                 }
    //             } catch (tf::TransformException ex) {
    //                 note << ex.what();
    //                 ROS_INFO_STREAM(ex.what());
    //                 servo_success = true;
    //                 return;
    //             }
    //
    //             lastErr = error;
    //
    //             sleep(wait_time);
    //         }
    //     }
    // }
    //
    void open_servo() {
        ROS_INFO_STREAM("Begin open-loop servoing");

        servo_success = false;

        while (!servo_success) {
            arm_pose = move_group.getCurrentPose();
            // secondaryImageScale = arm_pose.pose.position.z - CAMERA_OFFSET -
            // objectposz;
            // ROS_INFO_STREAM("secondaryImageScale: " << secondaryImageScale);
            // updateSendScale();

            // servoMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8",
            // grey).toImageMsg();
            servoMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color)
                           .toImageMsg();

            scale.data = {1.0,1.0};
            scalePub.publish(scale);
            servoPub.publish(servoMsg);
            reg_complete = false;

            while (!reg_complete) {
            }

            if (reg_success) {
                tf::Quaternion q(
                    arm_pose.pose.orientation.x, arm_pose.pose.orientation.y,
                    arm_pose.pose.orientation.z, arm_pose.pose.orientation.w);
                tf::Matrix3x3 m(q);
                double roll, pitch, yaw;
                m.getRPY(roll, pitch, yaw);

                arm_pose.pose.orientation =
                    tf::createQuaternionMsgFromRollPitchYaw(
                        roll, pitch, yaw - servo_rotation);
                // ROS_INFO_STREAM("obj estimate: " << objectposz);
                // objectposz = objectposz + (initialImageScale - initialImageScale*(1 +
                // (servo_scale-1)*initialImageMult/secondImageMult)) * (float(color.cols) /
                // regionOfInterest.cols);
                // ROS_INFO_STREAM("obj estimate updated: " << objectposz);
                ROS_INFO_STREAM("initialImageScale: " << initialImageScale);
                initialImageScale /=
                    servo_scale;  // (1 + (servo_scale-1)*initialImageMult/secondImageMult);
                ROS_INFO_STREAM("initialImageScale updated: " << initialImageScale);
                // arm_pose.pose.position.z = - objectposz + initialImageScale +
                // CAMERA_OFFSET; // (estimate of object position in world)
                arm_pose.pose.position.z =
                    initialImageScale +
                    CAMERA_OFFSET;  // (estimate of object position in world)
                ROS_INFO_STREAM("z: " << arm_pose.pose.position.z);
                float dx = (initialImageScale * tan(HALF_HFOV * M_PI / 180) * 2) *
                           (-servo_x / color.cols);  // (estimate of width of
                // image (width) contents)
                // * (translation in pixels
                // / image width in pixels)
                float dy = (initialImageScale * tan(HALF_HFOV * M_PI / 180) * 2) *
                           (servo_y / color.cols);
                ROS_INFO_STREAM("dx: " << dx);
                ROS_INFO_STREAM("dy: " << dy);
                arm_pose.pose.position.x -= dx;
                arm_pose.pose.position.y += dy;  // sign due to world axes

                ROS_INFO_STREAM("Moving to servo position");
                try {
                    if (!moveTo(arm_pose, 2, wait_time)) {
                        throw tf::TransformException(
                            "Move to servo position failed");
                    }
                } catch (tf::TransformException ex) {
                    note << ex.what();
                    ROS_INFO_STREAM(ex.what());
                    servo_success = true;
                    return;
                }

                sleep(wait_time);
            }
        }
    }
    //
    void servoCallback(const std_msgs::Float32MultiArray msg) {
        ROS_INFO_STREAM("SURF information received from MATLAB");
        servo_data = msg.data;
        Eigen::Map<Eigen::MatrixXf> mat(
            servo_data.data(), msg.layout.dim[0].size, msg.layout.dim[1].size);
        reg_success = true ? mat(0, 0) : false;
        if (reg_success) {
            ROS_INFO_STREAM("MATLAB returned: " << mat);
            servo_scale = mat(0, 1);
            servo_rotation = mat(0, 2);
            servo_x = -mat(0, 4);
            servo_y = mat(0, 3);
            servo_attempts = 10;
            if (abs(1 - servo_scale) < thresh[0] &&
                abs(servo_rotation) < thresh[1] * M_PI / 180 &&
                abs(servo_x) < thresh[2] && abs(servo_y) < thresh[3]) {
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

    void imageMsgToMat(const sensor_msgs::Image::ConstPtr msgImage,
                   cv::Mat& image) {
        cv_bridge::CvImageConstPtr pCvImage;
        pCvImage = cv_bridge::toCvShare(msgImage, "bgr8");
        pCvImage->image.copyTo(image);
    }

    static void no_click(int event, int x, int y, int, void* this_) {
        static_cast<SeqSLAM_TPP*>(this_)->no_click(event, x, y);
    }

    void no_click(int event, int x, int y) {}

    void mouse_click(int event, int x, int y) {
        switch (event) {
            case cv::EVENT_LBUTTONDOWN: {
                pointA.x = x;
                pointA.y = y;
                time(&timer);

                time_id = dtoi(difftime(timer, 0));
                recordMsg.data = std::to_string(time_id);
                recordPub.publish(recordMsg);
                break;
            }

            case cv::EVENT_LBUTTONUP: {
                pointB.x = x;
                pointB.y = y;

                displaySelectedRegion();

                // initialImageScale = move_group.getCurrentPose().pose.position.z -
                //               CAMERA_OFFSET - objectposz;
                //
                // watsonScaleFromRef = initialImageScale;
                //
                Froi = static_cast<float>(color.cols) / regionOfInterest.cols;
                //
                // initialImageScale /= Froi;
                //
                // watsonScaleFromROI = initialImageScale;

                regionSelected = true;
                break;
            }

            case cv::EVENT_MBUTTONDOWN: {
                //
                //   pointA.x = 0;
                //   pointA.y = 0;
                //   pointB.x = color.size().width;
                //   pointB.y = color.size().height;
                //
                //   displaySelectedRegion();
                //
                //   initialImageScale = move_group.getCurrentPose().pose.position.z -
                //   CAMERA_OFFSET - objectposz;
                //
                //   reset_pixl = true;
                break;
            }

            case cv::EVENT_RBUTTONDOWN: {

                pixlMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color)
                              .toImageMsg();

                color.copyTo(pixl_initial_image);

                secondaryImagePub.publish(pixlMsg);

                scalePub.publish(scale);

                regionSelected = true;

                // secondaryImageScale = move_group.getCurrentPose().pose.position.z -
                //             CAMERA_OFFSET - objectposz;
                //
                // updateSendScale();
                break;
            }
        }
    }

    static void mouse_click(int event, int x, int y, int, void* this_) {
        static_cast<SeqSLAM_TPP*>(this_)->mouse_click(event, x, y);
    }

    void displaySelectedRegion() {
        servoing = true;

        color(cv::Rect(pointA.x, pointA.y, pointB.x - pointA.x,
                       pointB.y - pointA.y))
            .copyTo(regionOfInterest);
        color.copyTo(watson_initial_image);
        cv::rectangle(watson_initial_image, pointA, pointB,
                      cv::Scalar(255, 255, 0), 2);
        cv::namedWindow(REGION_WINDOW, cv::WINDOW_NORMAL);

        cv::imshow(REGION_WINDOW, regionOfInterest);
        watsonMsg =
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", regionOfInterest)
                .toImageMsg();
        initialImagePub.publish(watsonMsg);
    }

    void pixl_click(int event, int x, int y) {
        switch (event) {
            case cv::EVENT_LBUTTONDOWN: {
                //   pixlMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8",
                //   grey).toImageMsg();
                //   grey.copyTo(pixl_initial_image);
                //   secondaryImagePub.publish(pixlMsg);
                //   scalePub.publish(scale);
                //   secondaryImageScale = objectposz +
                //   move_group.getCurrentPose().pose.position.z -
                //   CAMERA_OFFSET;
                //   HALF_HFOVdScale();
                break;
            }

            case cv::EVENT_MBUTTONDOWN: {
                reset_pixl = true;
                break;
            }

            case cv::EVENT_RBUTTONDOWN: {
                reset_watson = true;

                break;
            }
        }
    }

    static void pixl_click(int event, int x, int y, int, void* this_) {
        static_cast<SeqSLAM_TPP*>(this_)->pixl_click(event, x, y);
    }

    void displayFeatureRegion() {
        cv::Point rad(20, 20);

        regionOfInterest(cv::Rect(featurePoint - rad, featurePoint + rad))
            .copyTo(featureOfInterest);

        cv::namedWindow(FEATURE_WINDOW, cv::WINDOW_NORMAL);
        cv::imshow(FEATURE_WINDOW, featureOfInterest);
        cv::namedWindow(LO_RES_WINDOW, cv::WINDOW_NORMAL);
        // imageLRSub = nh_.subscribe("/camera_lores/image", 10,
        // &pixl::callbackLR, this);
        // imageLRSub = nh_.subscribe("/nothing", 10, &pixl::callbackLR, this);
        cv::setMouseCallback(LO_RES_WINDOW, pixl_click,
                             this);  // used for user input for moving between
        // watson and pixl positions
    }

    void save_images() {
        std::ostringstream s;
        s << "/home/james/Dropbox/NASA/experiment/robust_images/" << time_id
          << "_ref_initial.png";
        cv::imwrite(s.str(), watson_initial_image);
        s.str("");
        s.clear();
        s << "/home/james/Dropbox/NASA/experiment/robust_images/" << time_id
          << "_sensor_initial.png";
        cv::imwrite(s.str(), pixl_initial_image);
        s.str("");
        s.clear();
        s << "/home/james/Dropbox/NASA/experiment/robust_images/" << time_id
          << "_regofint.png";
        cv::imwrite(s.str(), regionOfInterest);
        s.str("");
        s.clear();
        s << "/home/james/Dropbox/NASA/experiment/robust_images/" << time_id
          << "_sensor_final.png";
        cv::imwrite(s.str(), color);
        s.str("");
        s.clear();
        s << "/home/james/Dropbox/NASA/experiment/robust_images/" << time_id
          << "_sensor_est.png";
        cv::imwrite(s.str(), pixl_est_image);
    }

    void save_result() {
        result_file.open(
            "/home/james/Dropbox/NASA/experiment/robust_results.csv",
            std::ios_base::app);
        save_images();
        // unique id
        result_file << time_id << ",";

        // successful?
        bool test = true ? operator_success == "y" : false;
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
        tf::Quaternion q(
            watson_pose.pose.orientation.x, watson_pose.pose.orientation.y,
            watson_pose.pose.orientation.z, watson_pose.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        result_file << roll << ",";
        result_file << pitch << ",";
        result_file << yaw << ",";

        // watson full image
        result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"
                    << time_id << "_ref_initial.png"
                    << ",";

        // region of interest image
        result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"
                    << time_id << "_regofint.png"
                    << ",";

        // PIXL position
        result_file << pixl_pose.pose.position.x << ",";
        result_file << pixl_pose.pose.position.y << ",";
        result_file << pixl_pose.pose.position.z << ",";
        tf::Quaternion q1(
            pixl_pose.pose.orientation.x, pixl_pose.pose.orientation.y,
            pixl_pose.pose.orientation.z, pixl_pose.pose.orientation.w);
        tf::Matrix3x3 m1(q1);
        m1.getRPY(roll, pitch, yaw);
        result_file << roll << ",";
        result_file << pitch << ",";
        result_file << yaw << ",";

        // PIXL full image
        result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"
                    << time_id << "_sensor_initial.png"
                    << ",";

        // estimated position
        result_file << est_pose.pose.position.x << ",";
        result_file << est_pose.pose.position.y << ",";
        result_file << est_pose.pose.position.z << ",";
        tf::Quaternion q2(
            est_pose.pose.orientation.x, est_pose.pose.orientation.y,
            est_pose.pose.orientation.z, est_pose.pose.orientation.w);
        tf::Matrix3x3 m2(q2);
        m2.getRPY(roll, pitch, yaw);
        result_file << roll << ",";
        result_file << pitch << ",";
        result_file << yaw << ",";

        // PIXL full image
        result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"
                    << time_id << "_sensor_est.png"
                    << ",";

        // servoed position
        result_file << servo_pose.pose.position.x << ",";
        result_file << servo_pose.pose.position.y << ",";
        result_file << servo_pose.pose.position.z << ",";
        tf::Quaternion q3(
            servo_pose.pose.orientation.x, servo_pose.pose.orientation.y,
            servo_pose.pose.orientation.z, servo_pose.pose.orientation.w);
        tf::Matrix3x3 m3(q3);
        m3.getRPY(roll, pitch, yaw);
        result_file << roll << ",";
        result_file << pitch << ",";
        result_file << yaw << ",";

        // final image
        result_file << "/home/james/Dropbox/NASA/experiment/robust_images/"
                    << time_id << "_sensor_final.png"
                    << ",";

        // record estimate of distance to sample surface
        result_file << initialImageScale << ",";

        // add any notes
        result_file << note.str() << ",";
        note.str("");
        note.clear();

        result_file << std::endl;
        result_file.close();
    }

    int dtoi(double d) { return ceil(d - 0.5); }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "seqslam_tpp");
    SeqSLAM_TPP seqslam_tpp;


    return 0;
}
