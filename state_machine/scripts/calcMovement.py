import rospy
import smach
from moveit_lib.srv import get_pose,get_poseRequest
import baxter_interface

import numpy as np

import time

from std_msgs.msg import String
from std_msgs.msg import Int32, Float32
import math

class CalcMovement(smach.State):

    #define HALF_HFOV 27  // half horizontal FOV of the camera
    #define HALF_VFOV 18  // half vertical FOV of the camera

    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed','aborted'],
                                   input_keys=['data'],output_keys=['data'])
        rospy.loginfo("[CalcMovement]: Ready")

    # ==========================================================
    def execute(self, userdata):

        try:

            userdata['goal_pose'] = userdata['data']['poses'][1]

            # CHECK THIS CALCULATION AGAINST PAPER
            initialImageScale = userdata['data']['scales'][0] / userdata['data']['registration_scale']
            mult = (initialImageScale * tan(HALF_HFOV * math.pi / 180) * 2) / userdata['data']['initial_image'].width

            dx = mult * userdata['data']['t_x'];
            dy = mult * userdata['data']['t_y'];

            quaternion = (userdata['goal_pose'].orientation.x,userdata['goal_pose'].orientation.y,userdata['goal_pose'].orientation.z,userdata['goal_pose'].orientation.w)

            euler = tf.transformations.euler_from_quaternion(quaternion)

            ind = ['x','y','z'].index(userdata['data']['sample_direction'])

            euler[ind] += userdata['data']['rotation']

            quaternion = tf.transformations.quaternion_from_euler(*euler)

            userdata['goal_pose'].position.x -= dx;
            userdata['goal_pose'].position.y += dy;

            userdata['goal_pose'].orientation.x = quaternion[0]
            userdata['goal_pose'].orientation.y = quaternion[1]
            userdata['goal_pose'].orientation.z = quaternion[2]
            userdata['goal_pose'].orientation.w = quaternion[3]

            return 'succeeded'

        except Exception, e:
            rospy.logerr('Some Exception in CalcMovement!')
            rospy.loginfo("[CalcMovement]: " + e)
            return 'failed'
