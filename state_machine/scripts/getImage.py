import rospy
from rospy import ServiceException, ROSException
import smach
from seqslam_tpp.srv import UserSelection,UserSelectionRequest

import cv_bridge

class GetImage(smach.State):

    count = 0

    def __init__(self, action='get_image',tag=None):
        smach.State.__init__(self, input_keys=['data'], output_keys=['data'],
                             outcomes=['succeeded', 'failed','aborted'])

        # wait for the service to appear
        rospy.loginfo('Waiting for user_input_request service to come up ...')

        self.action = action
        self.tag = tag
        self.cv_bridge = cv_bridge.CvBridge()
        srv_name = '/seqslam_tpp/get_image'

        try:
            rospy.wait_for_service(srv_name, timeout=1)
        except ROSException:
            rospy.logerr('Service of %s not available. Restart and try again.' % srv_name)

        self.service = rospy.ServiceProxy(srv_name, UserSelection)

    # ==========================================================
    def execute(self, userdata):

        self.count += 1

        try:
            response = self.service.call(UserSelectionRequest())

            if response.success.data:
                if self.action == 'get_image':
                    userdata['data'][self.tag] = response.image
                    if 'recorded_images' not in userdata['data'].keys():
                        userdata['data']['recorded_images'] = []
                    image = self.cv_bridge.imgmsg_to_cv2(response.image)
                    userdata['data']['recorded_images'] = userdata['data']['recorded_images'] + [image,]

                return 'succeeded'

            else:
                rospy.logwarn('[UserInputRequest]: ' + response.message.data)
                raise NameError('ServiceFailed')

        except (NameError, ServiceException) as e:

            rospy.logwarn('[UserInputRequest]: ' + e.strerror)

            if self.count < userdata['data']['max_count']:
                return 'failed'
            else:
                self.count = 0
                return 'aborted'
