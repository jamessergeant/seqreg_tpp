#from spaceBot import *
import rospy
import smach
import smach_ros
import threading

import pickle as P

import time

current_milli_time = lambda: int(round(time.time() * 1000))

class ToggleData(smach.State):
    def __init__(self, key=None, value=None):
        smach.State.__init__(self, outcomes=['succeeded'], #, 'failed'],
                             input_keys=['data'],output_keys=['data']) # suctionState needs that one
        self.key = key
        self.value = value
        print "[ToggleData]: Ready"


    # ==========================================================
    def execute(self, userdata):

        try:
            if self.key is not None:
                if self.value is not None:
                    userdata['data'][self.key] = self.value
                else:
                    userdata['data'][self.key] = not userdata['data'][self.key]

            return 'succeeded'
        except:
            return 'failed'
