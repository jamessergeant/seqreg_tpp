import rospy
from rospy import ServiceException, ROSException
import smach
from seqslam_tpp.srv import MATLABSrv,MATLABSrvRequest,MATLABSrvResponse
from std_msgs.msg import Float32MultiArray, Empty
from sensor_msgs.msg import Image

import pickle as P

import time

current_milli_time = lambda: int(round(time.time() * 1000))

class SeqSLAMState(smach.State):

    count = 0

    def __init__(self, action='initial_position',method='ros_seqslam',save_msg=True):
        smach.State.__init__(self, input_keys=['data'], output_keys=['data'],
                             outcomes=['succeeded','failed','aborted'])

        self.action = action
        self.method = method
        self.save_msg = save_msg

        if self.action == 'initial_position':
            srv_name = '/seqslam_tpp/seqslam'
        elif self.action == 'servoing':
            srv_name = '/seqslam_tpp/seqslam_servo'
        rospy.loginfo('Waiting for ' + srv_name + ' service to come up ...')

        try:
            rospy.wait_for_service(srv_name, timeout=1)
        except ROSException:
            rospy.logerr('Service of %s not available. Restart and try again.' % srv_name)

        self.service = rospy.ServiceProxy(srv_name, MATLABSrv)

    # ==========================================================
    def execute(self, userdata):

        self.count += 1

        try:
            request = MATLABSrvRequest()

            request.initial_image = userdata['data']['roi']
            request.secondary_image = userdata['data']['secondary_image']
            request.scales.data = userdata['data']['relative_scales']
            request.method.data = self.method

            if self.save_msg:
                P.dump(request,open('/home/james/Dropbox/NASA/test_msgs/%i.pkl' % current_milli_time(),'wb'))

            response = self.service.call(request)

            if response.success.data:

                userdata['data']['registration_scale'] = response.results.data[0]
                userdata['data']['rotation'] = response.results.data[1]
                userdata['data']['t_x'] = response.results.data[2]
                userdata['data']['t_y'] = response.results.data[3]
                rospy.logwarn("[SeqSLAMState]: Trans X: %i" % userdata['data']['t_x'])
                rospy.logwarn("[SeqSLAMState]: Trans Y: %i" % userdata['data']['t_y'])

                if 'results' not in userdata['data'].keys():
                    userdata['data']['results'] = []

                userdata['data']['results'] = userdata['data']['results'] + [response.results.data,]
                userdata['data']['relative_scales'] = [1,1]

                return 'succeeded'

            else:
                rospy.logwarn('[SeqSLAMState]: SeqSLAM registration failed.')
                rospy.logwarn('[SeqSLAMState]: Service Message: ' + response.message.data)
                return 'aborted'

        except ServiceException as e:
            rospy.logwarn('[SeqSLAMState]: ' + str(e))

            if self.count < userdata['data']['max_count']:
                return 'failed'
            else:
                self.count = 0
                return 'aborted'
