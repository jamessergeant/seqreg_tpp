#from spaceBot import *
import rospy
import smach
import smach_ros
import threading


class AbortState(smach.State):
    def __init__(self, duration=1.0):
        smach.State.__init__(self, outcomes=['succeeded'], #, 'failed'],
                             input_keys=['move_group']) # suctionState needs that one
        print "[AbortState]: Ready"


    # ==========================================================
    def execute(self, userdata):
        # wait for the specified duration or until we are asked to preempt
        rospy.signal_shutdown('[AbortState]: Who knows?')
        return 'succeeded'
