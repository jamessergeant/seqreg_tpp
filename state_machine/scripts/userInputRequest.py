import rospy
from rospy import ServiceException, ROSException
import smach
from seqslam_tpp.srv import UserSelection,UserSelectionRequest

class UserInputRequest(smach.State):

    count = 0

    def __init__(self, action='user_request_initial'):
        smach.State.__init__(self, input_keys=['data'], output_keys=['data'],
                             outcomes=['succeeded', 'failed','abort'])

        # wait for the service to appear
        rospy.loginfo('Waiting for user_input_request service to come up ...')

        self.action = action
        srv_name = '/seqslam_tpp/user_input_request'

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
                if self.action == 'user_request_initial':
                    userdata['data']['initial_image'] = response.image
                    userdata['data']['roi'] = response.roi
                    userdata['data']['bounding_box'] = response.bounding_box
                # if self.action == 'user_request_secondary':
                #     userdata['data']['secondary_image'] = response.image
                #     userdata['data']['scales'] = response.scales

                return 'succeeded'

            else:
                rospy.logwarn('[UserInputRequest]: ' + response.message.data)
                raise NameError('ServiceFailed')

        except (NameError, ServiceException) as e:

            rospy.logwarn('[UserInputRequest]: ' + e.strerror)

            if self.count < userdata['max_count']:
                return 'failed'
            else:
                self.count = 0
                return 'abort'
