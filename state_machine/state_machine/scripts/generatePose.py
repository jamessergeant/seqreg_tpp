import rospy
import smach
import random

class GeneratePose(smach.State):
    def __init__(self, action=None, pose=None,noise=None,tag=None):
        smach.State.__init__(self, output_keys=['pose'],
                             outcomes=['succeeded', 'failed'])

        rospy.loginfo('Generating pose.')

        self.action = action # not used currently, just a placeholder in case
        self.pose = pose
        self.noise = noise
        self.tag = tag

    # ==========================================================
    def execute(self, userdata):

        random.seed()

        pose = self.pose

        pose.position.x += random.normalvariate(0.0, self.noise[0])
        pose.position.y += random.normalvariate(0.0, self.noise[1])
        pose.position.z += random.normalvariate(0.0, self.noise[2])

        userdata = {'pose': pose,'data': {tag: pose}}
        userdata = {'pose': pose,'data': {tag: pose}}

        if self.tag is not None:
            userdata = {'pose': pose,'data': {tag: pose}}
        else:
            userdata = {'pose': pose}

        return 'succeeded'
