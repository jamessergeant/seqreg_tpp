import rospy
import smach
import random
from copy import deepcopy
import pickle as pkl

class GeneratePose(smach.State):
    def __init__(self, action=None, pose=None,frame_id=None,noise=None,tag=None,path=None):
        smach.State.__init__(self, input_keys=['data'],output_keys=['goal_frame_id','goal_pose','data'],
                             outcomes=['succeeded', 'failed'])

        rospy.loginfo('Generating pose.')

        self.action = action # not used currently, just a placeholder in case
        self.pose_ = pose
        self.frame_id = frame_id
        self.noise = noise
        self.tag = tag
        self.path = path

    # ==========================================================
    def execute(self, userdata):

        if self.action is None:
            random.seed()

            pose = deepcopy(self.pose_)
            pose.position.x += random.normalvariate(0.0, self.noise[0])
            pose.position.y += random.normalvariate(0.0, self.noise[1])
            pose.position.z += random.normalvariate(0.0, self.noise[2])

            if 'data' not in userdata.keys():
                userdata['data'] = {}

            userdata['data'][self.tag] = pose
            userdata['data']['frame_id'] = self.frame_id
            userdata['data']['distance_estimate'] = userdata['data']['sample_distance'] + random.normalvariate(0.0,
                                                        userdata['data']['sample_noise'])

            if 'output' not in userdata['data'].keys():
                userdata['data']['output'] = {}
            userdata['data']['output']['distance_estimate'] = userdata['data']['distance_estimate']

            userdata['goal_pose'] = pose
            userdata['goal_frame_id'] = self.frame_id
            userdata['data']['goal_frame_id'] = self.frame_id

            print userdata

            return 'succeeded'
        elif self.action == 'load_existing':
            random.seed()
            if 'data' not in userdata.keys():
                userdata['data'] = {}
            data = pkl.load(open(self.path,'rb'))
            userdata['data'][self.tag] = data['recorded_poses'][0]
            userdata['data']['frame_id'] = self.frame_id
            userdata['data']['distance_estimate'] = userdata['data']['sample_distance'] + random.normalvariate(0.0,
                                                        userdata['data']['sample_noise'])

            if 'output' not in userdata['data'].keys():
                userdata['data']['output'] = {}
            userdata['data']['output']['distance_estimate'] = userdata['data']['distance_estimate']

            userdata['goal_pose'] = data['recorded_poses'][0]
            userdata['goal_frame_id'] = self.frame_id
            userdata['data']['goal_frame_id'] = self.frame_id
            return 'succeeded'
