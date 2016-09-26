#!/usr/bin/env python
import rospy
import time
import pickle
from seqslam_tpp.srv import MATLABSrv

class test_node:

    def __init__(self):
        rospy.wait_for_service('/seqslam_tpp/seqslam', timeout=1)

        self.service = rospy.ServiceProxy('/seqslam_tpp/seqslam', MATLABSrv)

        # request = pickle.load(open('/home/james/Dropbox/NASA/test_msgs/1473211892068.pkl','rb'))
        # request = pickle.load(open('/home/james/Dropbox/NASA/test_msgs/1471565941813.pkl','rb'))
        request = pickle.load(open('/home/james/Dropbox/NASA/test_msgs/1473211993758.pkl','rb'))

        time.sleep(2)

        response = self.service.call(request)


def main():

  rospy.init_node('test_node', anonymous=True)
  TN = test_node()

  # rospy.spinOnce()

if __name__ == '__main__':

    main()
