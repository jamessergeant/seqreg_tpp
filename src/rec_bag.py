#!/usr/bin/env python
from subprocess import Popen
from std_msgs.msg import String
import rospy
import os
import signal

class bag_record:

    def __init__(self):
        self.recording = False
        self.recordSub = rospy.Subscriber("/pixl/record", String, self.record)


    def record(self,msg):

        if not self.recording:
            self.recording = True
            self.recorder = Popen('rosbag record -a -O /home/james/Dropbox/NASA/experiment/robust_bags/%s.bag' % msg.data, shell=True, preexec_fn=os.setsid)
        else:
            self.recording = False
            self.recorder.send_signal(signal.SIGINT)
            os.killpg(self.recorder.pid,signal.SIGTERM)


    def shutdown(self):
        os.killpg(self.recorder.pid, signal.SIGTERM)

def main():

  rospy.init_node('bag_record', anonymous=True)
  BR = bag_record()

  rospy.on_shutdown(BR.shutdown)

  rospy.spin()

if __name__ == '__main__':

    main()
