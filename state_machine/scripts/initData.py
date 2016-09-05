#from spaceBot import *
import rospy
import smach
import smach_ros
import threading


class InitData(smach.State):
    def __init__(self, sample_distance, sample_direction, sample_noise, limits, camera_information):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'],
                             input_keys=['data'],output_keys=['data']) # suctionState needs that one
        self.sample_distance = sample_distance
        self.sample_direction = sample_direction
        self.sample_noise = sample_noise
        self.limits = limits
        self.camera_information = camera_information
        print "[InitData]: Ready"


    # ==========================================================
    def execute(self, userdata):

        # TODO save userdata
        userdata['data'] = {}
        userdata['data']['max_count'] = 5
        userdata['data']['sample_distance'] = self.sample_distance
        userdata['data']['sample_direction'] = self.sample_direction
        userdata['data']['sample_noise'] = self.sample_noise
        userdata['data']['limits'] = self.limits
        userdata['data']['camera_information'] = self.camera_information
        userdata['data']['servoing'] = False

        return 'succeeded'
