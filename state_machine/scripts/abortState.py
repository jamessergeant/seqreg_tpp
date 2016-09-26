#from spaceBot import *
import rospy
import smach
import smach_ros
import threading

import pickle as P

import time

current_milli_time = lambda: int(round(time.time() * 1000))

class AbortState(smach.State):
    def __init__(self, path='/home/james/Dropbox/NASA/baxter_experiments/userdata'):
        smach.State.__init__(self, outcomes=['succeeded'], #, 'failed'],
                             input_keys=['data'],output_keys=['data']) # suctionState needs that one
        self.path = path
        print "[AbortState]: Ready"


    # ==========================================================
    def execute(self, userdata):

        try:
            # TODO save userdata
            timestamp = current_milli_time()
            userdata['data']['output']['poses'] = userdata['data']['poses']
            P.dump(userdata['data']['output'], open('%s/%i.pkl' % (self.path,timestamp), 'wb'))

            return 'succeeded'
        except:
            return 'failed'
