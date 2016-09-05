import rospy
import smach
from moveit_lib.srv import get_pose,get_poseRequest
import baxter_interface

import numpy as np

import time

from std_msgs.msg import String
from std_msgs.msg import Int32, Float32

class GetRobotPose(smach.State):

    def __init__(self, action='get_pose',movegroup=None,frame_id=None,calculate_scales=False):
        smach.State.__init__(self, outcomes=['succeeded', 'failed','aborted'],
                                   input_keys=['data'],output_keys=['data'])

        if None in [movegroup,frame_id]:
            rospy.logwarn("Parameter movegroup or frame_id not specified. GetRobotPose instance will not execute.")

        self.action = action
        self.movegroup = movegroup
        self.frame_id = frame_id
        self.calculate_scales = calculate_scales

        service_name = '/moveit_lib/get_pose'
        # wait for the service to appear
        rospy.loginfo('Waiting for service %s to come online ...' % service_name)
        try:
            rospy.wait_for_service(service_name, timeout=0.1)
        except:
            rospy.logerr('Service %s not available. Restart and try again.' % service_name)

        self.service = rospy.ServiceProxy(service_name,
                                    get_pose)

    # ==========================================================
    def execute(self, userdata):

        try:

            if self.movegroup is None:
                rospy.logerr("Parameter movegroup is not specified. Aborting.")
                return 'failed'

            request = get_poseRequest()
            request.move_group.data = self.movegroup
            request.frame_id.data = self.frame_id
            response = self.service.call(request)
            rospy.loginfo(response.pose.pose)

            if response.success.data:

                rospy.loginfo('[GetPose]: Pose in frame: ' + response.pose.header.frame_id + ' for movegroup: ' + self.movegroup)

                if 'poses' not in userdata['data']:
                    userdata['data']['poses'] = []
                elif len(userdata['data']['poses']) == 2:
                    userdata['data']['poses'].pop(1)

                userdata['data']['poses'] = userdata['data']['poses'] + [response.pose.pose,]

                if 'output' not in userdata['data'].keys():
                    userdata['data']['output'] = {}
                if 'recorded_poses' not in userdata['data']['output'].keys():
                    userdata['data']['output']['recorded_poses'] = []

                userdata['data']['output']['recorded_poses'] = userdata['data']['output']['recorded_poses'] + [response.pose.pose,]

                rospy.loginfo(userdata['data']['poses'])

                return 'succeeded'
            else:
                return 'failed'

        except Exception, e:
            rospy.logerr('Some Exception in GetRobotPose!')
            print e
            return 'aborted'
