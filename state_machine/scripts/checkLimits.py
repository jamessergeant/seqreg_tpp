import rospy
import smach
from moveit_lib.srv import get_pose,get_poseRequest
import baxter_interface

import numpy as np

import time

from std_msgs.msg import String
from std_msgs.msg import Int32, Float32
import math

class CheckLimits(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['incomplete', 'complete','aborted'],
                                   input_keys=['data'],output_keys=['data'])
        rospy.loginfo("[CheckLimits]: Ready")

    # ==========================================================
    def execute(self, userdata):

        try:
            results = [abs(1 - userdata['data']['results'][-1][0]),
                abs(userdata['data']['results'][-1][1]) * math.pi / 180,
                abs(userdata['data']['results'][-1][2]),
                abs(userdata['data']['results'][-1][3])]

            print results
            print (np.asarray(results) > np.asarray(userdata['data']['limits']))
            if (np.asarray(results) > np.asarray(userdata['data']['limits'])).any():
                return 'incomplete'

            return 'complete'

        except Exception, e:
            rospy.logerr('Some Exception in CheckLimits!')
            rospy.loginfo("[CheckLimits]: " + e.message)
            return 'aborted'
