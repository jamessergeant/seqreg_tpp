import rospy
import smach
from moveit_lib.srv import get_pose,get_poseRequest
import baxter_interface

import numpy as np

import time

from std_msgs.msg import String
from std_msgs.msg import Int32, Float32
import math
import tf.transformations
from geometry_msgs.msg import Quaternion

class CalcMovement(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed','aborted'],
                                   input_keys=['data'],output_keys=['data','goal_pose','goal_frame_id'])
        rospy.loginfo("[CalcMovement]: Ready")


    def shift(self, seq, n):
        n = n % len(seq)
        return seq[n:] + seq[:n]

    # ==========================================================
    def execute(self, userdata):

        try:

            goal_pose = userdata['data']['poses'][1]

            rospy.logwarn("[CalcMovement]: Moving From: ")
            rospy.logwarn(goal_pose)

            ind = ['x','y','z'].index(userdata['data']['sample_direction'])

            current_position = [goal_pose.position.x,goal_pose.position.y,goal_pose.position.z][ind]

            # CALCULATE TRANSLATION BASED ON REGISTRATION OFFSET AND SCALE
            estimated_offset = userdata['data']['estimated_offset'] / userdata['data']['registration_scale']

            rospy.logwarn("[CalcMovement]: estimated_offset: %0.3f" % estimated_offset)
            rospy.logwarn("[CalcMovement]: Trans X: %0.3f" % userdata['data']['t_x'])
            rospy.logwarn("[CalcMovement]: Trans Y: %0.3f" % userdata['data']['t_y'])

            # rospy.logwarn("[CalcMovement]: registration_scale %0.3f" % userdata['data']['registration_scale'])
            # rospy.logwarn("[CalcMovement]: orig_scale %0.3f" % userdata['data']['scales'][0])
            # rospy.logwarn("[CalcMovement]: roi_scale %0.3f" % userdata['data']['roi_scale'])
            # rospy.logwarn("[CalcMovement]: half_hfov %i" % userdata['data']['camera_information']['half_hfov'])

            current_offset = userdata['data']['distance_estimate'] - current_position

            # 400 used as max width in the seqslam algorithm
            meters_per_pixel = 2 * (current_offset / 400) * math.tan(userdata['data']['camera_information']['half_hfov'] * math.pi / 180)

            rospy.logwarn("[CalcMovement]: meters/pixel %0.5f" % meters_per_pixel)

            # print "distance_estimate: ", userdata['data']['distance_estimate']
            # print "scales[0]: ", userdata['data']['scales'][0]
            # print "registration_scale: ", userdata['data']['registration_scale']
            # print "current x position: ", [goal_pose.position.x,goal_pose.position.y,goal_pose.position.z][ind]

            dx = userdata['data']['distance_estimate'] - (estimated_offset + current_position)
            dy = meters_per_pixel * userdata['data']['t_y']
            dz = meters_per_pixel * userdata['data']['t_x']

            differences = self.shift([dx,dy,dz],ind)

            # UNCERTAIN OF SIGNS, depends of orientation of camera frame in reference frame
            goal_pose.position.x += differences[0]
            goal_pose.position.y -= differences[1]
            goal_pose.position.z -= differences[2]

            userdata['data']['estimated_offset'] = estimated_offset

            # CALCULATE ROTATION FROM REGISTRATION ROTATION
            quaternion = (goal_pose.orientation.x,
                            goal_pose.orientation.y,
                            goal_pose.orientation.z,
                            goal_pose.orientation.w)

            euler = tf.transformations.euler_from_quaternion(quaternion)
            euler = list(euler)
            euler[ind] += userdata['data']['rotation'] * math.pi / 180
            euler = tuple(euler)

            quaternion = tf.transformations.quaternion_from_euler(*euler)

            goal_pose.orientation = Quaternion(*(0,0,0,1))

            userdata['goal_frame_id'] = userdata['data']['goal_frame_id']
            userdata['goal_pose'] = goal_pose

            rospy.logwarn("[CalcMovement]: Moving To: ")
            rospy.logwarn(goal_pose)

            return 'succeeded'

        except Exception, e:
            rospy.logerr('Some Exception in CalcMovement!')
            rospy.loginfo("[CalcMovement]: " + e)
            return 'failed'
