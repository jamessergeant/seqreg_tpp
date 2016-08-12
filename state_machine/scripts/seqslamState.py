import rospy
from rospy import ServiceException, ROSException
import smach
from seqslam_tpp.srv import MATLABSrv,MATLABSrvRequest,MATLABSrvResponse
from std_msgs.msg import Float32MultiArray, Empty
from sensor_msgs.msg import Image

class SeqSLAMState(smach.State):

    count = 0

    def __init__(self, action='initial_position'):
        smach.State.__init__(self, input_keys=['data'], output_keys=['data'],
                             outcomes=['succeeded','failed','aborted'])

        # wait for the service to appear

        self.action = action

        # if self.action == 'initial_position':
        #     srv_name = '/seqslam_tpp/seqslam'
        # elif self.action == 'servoing':
        #     srv_name = '/seqslam_tpp/seqslam_servo'
        # rospy.loginfo('Waiting for ' + srv_name + ' service to come up ...')

        image1_srv = '/seqslam_tpp/image1'
        image2_srv = '/seqslam_tpp/image2'
        scales_srv = '/seqslam_tpp/scales'
        seqslam_srv = '/seqslam_tpp/seqslam'

        # try:
        #     # rospy.wait_for_service(srv_name, timeout=1)
        #     rospy.wait_for_service(image1_srv, timeout=1)
        #     rospy.wait_for_service(image1_srv, timeout=1)
        #     rospy.wait_for_service(scales_srv, timeout=1)
        # except ROSException:
        #     rospy.logerr('Service of %s not available. Restart and try again.' % srv_name)

        # self.service = rospy.ServiceProxy(srv_name, MATLABSrv)
        self.pub_image1 = rospy.Publisher(image1_srv, Image)
        self.pub_image2 = rospy.Publisher(image2_srv, Image)
        self.pub_scales = rospy.Publisher(scales_srv, Float32MultiArray)
        self.pub_seqslam = rospy.Publisher(seqslam_srv, Empty)

        self.sub_results = rospy.Subscriber('/seqslam_tpp/results',Float32MultiArray,self.resultsCallback)
        self.results_rec = False

    def resultsCallback(self,msg):
        rospy.loginfo('Results recieved')
        self.response = MATLABSrvResponse()
        self.response.results = msg
        self.results_rec = True

    # ==========================================================
    def execute(self, userdata):
        rospy.loginfo('[SeqSLAMState]: USERDATA')
        rospy.loginfo(userdata['data']['relative_scales'])
        self.count += 1

        try:
            request = MATLABSrvRequest()

            request.initial_image = userdata['data']['initial_image']
            request.secondary_image = userdata['data']['secondary_image']
            request.scales.data = userdata['data']['relative_scales']

            # response = self.service.call(request)

            self.pub_image1.publish(request.initial_image)
            self.pub_image2.publish(request.secondary_image)
            self.pub_scales.publish(request.scales)
            self.pub_seqslam.publish(Empty())

            while self.results_rec is False:
                donothing = 0
            self.results_rec = False
            response = self.response
            # if response.success.data:
            if True:
                userdata['data']['registration_scale'] = response.results.data[0]
                userdata['data']['rotation'] = response.results.data[1]
                userdata['data']['t_x'] = response.results.data[2]
                userdata['data']['t_y'] = response.results.data[3]

                if 'results' not in userdata['data'].keys():
                    userdata['data']['results'] = []

                userdata['data']['results'] = userdata['data']['results'] + [response.results.data,]

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
