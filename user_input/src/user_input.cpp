/*
Copyright 2016 Australian Centre for Robotic Vision
Author: James Sergeant james.sergeant@qut.edu.au
*/

#include <user_input/UserSelection.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_listener.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <vector>


static const std::string HI_RES_WINDOW = "Camera Image";
static const std::string REGION_WINDOW = "Selected Region Of Interest";

class UserInput {
    ros::NodeHandle nh_;
    tf::TransformListener tf_listener;

    ros::Subscriber imageSub;

    ros::ServiceServer userInputRequestSrv;
    ros::ServiceServer getImageSrv;

    sensor_msgs::Image imgMsg;

    cv::Point pointA;
    cv::Point pointB;

    std::string image_topic;

    bool regionSelected= false;

    bool live_image = true;

    float Froi;

    cv::Mat color;
    cv::Mat static_image;
    cv::Mat watson_initial_image;
    cv::Mat regionOfInterest;

    XmlRpc::XmlRpcValue poses;

   public:
    UserInput() {
        cv::namedWindow(HI_RES_WINDOW, cv::WINDOW_NORMAL);

        // obtain parameters
        nh_.param("/user_input/image_topic", image_topic,
                  std::string("/realsense/rgb/image_raw"));

        imageSub =
            nh_.subscribe(image_topic, 10, &UserInput::imageCallback, this);

        // Start Services
        userInputRequestSrv = nh_.advertiseService(
            "/user_input/user_input_request", &UserInput::userInputRequestCallback, this);
        getImageSrv = nh_.advertiseService(
            "/user_input/get_image", &UserInput::getImageCallback, this);

        ROS_INFO_STREAM("Init complete");

        ros::AsyncSpinner spinner(2);
        spinner.start();

        ros::waitForShutdown();
    }

    ~UserInput() {
        cv::destroyWindow(HI_RES_WINDOW);
        cv::destroyWindow(REGION_WINDOW);
    }


    bool userInputRequestCallback(user_input::UserSelection::Request &req,
                            user_input::UserSelection::Response &res) {
        if (!req.image.data.empty()) {
          cv_bridge::CvImageConstPtr pCvImage;
          pCvImage = cv_bridge::toCvCopy(req.image, "bgr8");
          pCvImage->image.copyTo(static_image);
          imgMsg = req.image;
          live_image = false;

        } else {
          live_image = true;
        }

        cv::setMouseCallback(HI_RES_WINDOW, mouse_click, this);

        while (!regionSelected) {
          // poor handling - consider other options
        }

        try {
            res.image = imgMsg;
            res.roi = *(cv_bridge::CvImage(std_msgs::Header(),
                            "bgr8", regionOfInterest).toImageMsg());

            res.roi_scale.data = Froi;

            // res.bounding_box.top_left.x = pointA.x;
            // res.bounding_box.top_left.y = pointA.y;
            // res.bounding_box.bottom_right.x = pointB.x;
            // res.bounding_box.bottom_right.y = pointB.y;

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

        live_image = true;
        return true;
    }

    bool getImageCallback(user_input::UserSelection::Request &req,
                            user_input::UserSelection::Response &res) {
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
      if (live_image) {
        imgMsg = *imageColor;
        imageMsgToMat(imageColor, color);
      } else {
            color = static_image;
      }
      cv::imshow(HI_RES_WINDOW, color);

      cv::waitKey(3);
    }

    void imageMsgToMat(const sensor_msgs::Image::ConstPtr msgImage,
                   cv::Mat& image) {
        cv_bridge::CvImageConstPtr pCvImage;
        pCvImage = cv_bridge::toCvShare(msgImage, "bgr8");
        pCvImage->image.copyTo(image);
    }


    static void no_click(int event, int x, int y, int, void* this_) {
        static_cast<UserInput*>(this_)->no_click(event, x, y);
    }

    void no_click(int event, int x, int y) {}

    void mouse_click(int event, int x, int y) {
        switch (event) {
            case cv::EVENT_LBUTTONDOWN: {
                pointA.x = x;
                pointA.y = y;

                break;
            }

            case cv::EVENT_LBUTTONUP: {
                pointB.x = x;
                pointB.y = y;

                displaySelectedRegion();

                Froi = static_cast<float>(color.cols) / regionOfInterest.cols;

                regionSelected = true;
                break;
            }

            case cv::EVENT_MBUTTONDOWN: {
                // Nothing implemented
                break;
            }
        }
    }

    static void mouse_click(int event, int x, int y, int, void* this_) {
        static_cast<UserInput*>(this_)->mouse_click(event, x, y);
    }

    void displaySelectedRegion() {
        color(cv::Rect(pointA.x, pointA.y, pointB.x - pointA.x,
                       pointB.y - pointA.y))
            .copyTo(regionOfInterest);
        color.copyTo(watson_initial_image);
        cv::rectangle(watson_initial_image, pointA, pointB,
                      cv::Scalar(255, 255, 0), 2);
        cv::namedWindow(REGION_WINDOW, cv::WINDOW_NORMAL);

        cv::imshow(REGION_WINDOW, regionOfInterest);
    }

    int dtoi(double d) { return ceil(d - 0.5); }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "user_input");
    UserInput user_input;

    return 0;
}
