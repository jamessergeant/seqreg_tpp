import rospy
import smach
from moveit_lib.srv import get_pose,get_poseRequest
import baxter_interface

import numpy as np

import time

from std_msgs.msg import String
from std_msgs.msg import Int32, Float32

class CalcScales(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed','aborted'],
                                   input_keys=['data'],output_keys=['data'])
        rospy.loginfo("[CalcScales]: Ready")

    # ==========================================================
    def execute(self, userdata):

        try:

            position_initial = {'x':userdata['data']['poses'][0].position.x,
                                'y':userdata['data']['poses'][0].position.y,
                                'z':userdata['data']['poses'][0].position.z,
            }
            position_secondary = {'x':userdata['data']['poses'][1].position.x,
                                'y':userdata['data']['poses'][1].position.y,
                                'z':userdata['data']['poses'][1].position.z,
            }

            rospy.logwarn(userdata['data']['distance_estimate'])

            scales = np.asarray([abs((position_initial[userdata['data']['sample_direction']]-userdata['data']['distance_estimate'])/userdata['data']['roi_scale']), abs(position_secondary[userdata['data']['sample_direction']]-userdata['data']['distance_estimate'])])

            rospy.loginfo(scales)

            if userdata['data']['servoing']:
                userdata['data']['relative_scales'] = [1,1]
            else:
                userdata['data']['scales'] = list(scales)
                userdata['data']['relative_scales'] = list(scales / float(scales.max()))
                userdata['data']['estimated_offset'] = userdata['data']['scales'][0]

            rospy.logwarn(userdata['data']['scales'])
            rospy.logwarn(userdata['data']['relative_scales'])

            return 'succeeded'

        except Exception, e:
            rospy.logerr('Some Exception in CalcScales!')
            rospy.loginfo("[CalcScales]: " + e)
            return 'failed'
