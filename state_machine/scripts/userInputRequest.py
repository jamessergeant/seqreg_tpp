import rospy
from rospy import ServiceException, ROSException
import smach
from seqreg_tpp.srv import UserSelection,UserSelectionRequest
from sensor_msgs.msg import Image
import pickle as pkl

import cv_bridge

class UserInputRequest(smach.State):

    count = 0

    def __init__(self, action='user_request_initial', path=None):
        smach.State.__init__(self, input_keys=['data'], output_keys=['data'],
                             outcomes=['succeeded', 'failed','abort','same_reference_image'])

        # wait for the service to appear
        rospy.loginfo('Waiting for user_input_request service to come up ...')

        self.action = action
        if self.action == 'user_request_initial' or self.action == 'user_reuse_initial' or self.action == 'load_existing':
            self.cv_bridge = cv_bridge.CvBridge()
            srv_name = '/seqreg_tpp/user_input_request'
            try:
                rospy.wait_for_service(srv_name, timeout=1)
            except ROSException:
                rospy.logerr('Service of %s not available. Restart and try again.' % srv_name)

            self.service = rospy.ServiceProxy(srv_name, UserSelection)
            if self.action == 'load_existing':
                assert(path is not None)
                self.path = path
        elif self.action == 'wait_for_key_press':
            print 'wait for key press'

    # ==========================================================
    def execute(self, userdata):
        if self.action == 'user_request_initial' or self.action == 'user_reuse_initial' or self.action == 'load_existing':
            self.count += 1

            try:
                request = UserSelectionRequest()
                request.image = Image()
                if self.action == 'user_reuse_initial':
                    request.image = userdata['data']['initial_image']

                elif self.action == 'load_existing':
                    data = pkl.load(open(self.path,'rb'))
                    request.image = self.cv_bridge.cv2_to_imgmsg(data['recorded_images'][0],'bgr8')
                    if 'poses' not in userdata['data']:
                        userdata['data']['poses'] = []
                    userdata['data']['poses'] = [data['recorded_poses'][0],]
                    if 'output' not in userdata['data'].keys():
                        userdata['data']['output'] = {}
                    if 'recorded_poses' not in userdata['data']['output'].keys():
                        userdata['data']['output']['recorded_poses'] = []

                    userdata['data']['output']['recorded_poses'] = userdata['data']['output']['recorded_poses'] + [data['recorded_poses'][0],]
                response = self.service.call(request)

                if response.success.data:
                    if self.action == 'user_request_initial' or self.action == 'user_reuse_initial' or self.action == 'load_existing':
                        userdata['data']['initial_image'] = response.image
                        userdata['data']['roi'] = response.roi
                        userdata['data']['roi_scale'] = response.roi_scale.data
                        rospy.logwarn('[UserInputRequest]: ROI Scale %0.3f' % userdata['data']['roi_scale'])
                        if 'output' not in userdata['data'].keys():
                            userdata['data']['output'] = {}
                        if 'recorded_images' not in userdata['data']['output'].keys():
                            userdata['data']['output']['recorded_images'] = []
                        image = self.cv_bridge.imgmsg_to_cv2(response.image)
                        roi = self.cv_bridge.imgmsg_to_cv2(userdata['data']['roi'])
                        userdata['data']['output']['recorded_images'] = userdata['data']['output']['recorded_images'] + [image,]
                        userdata['data']['output']['roi'] = roi

                    return 'succeeded'

                else:
                    rospy.logwarn('[UserInputRequest]: ' + response.message.data)
                    raise NameError('ServiceFailed')

            except NameError as e:

                rospy.logwarn('[UserInputRequest]: ' + e.strerror)

                if self.count < userdata['data']['max_count']:
                    return 'failed'
                else:
                    self.count = 0
                    return 'abort'
            except ServiceException as e:
                rospy.logwarn('[UserInputRequest]: ' + e)

                if self.count < userdata['data']['max_count']:
                    return 'failed'
                else:
                    self.count = 0
                    return 'abort'
        elif self.action == 'wait_for_key_press':
            rospy.loginfo('Waiting for keypress....')
            output = raw_input()

            if output == 'c': # continue
                return 'succeeded'

            elif output == 'n': # new trial
                return 'abort'

            elif output == 's': # new trial, same reference image
                return 'same_reference_image'

            else:
                return 'succeeded'
