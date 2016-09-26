#!/usr/bin/env python
import os
import rospy
import random
import glob
from std_msgs.msg import String
import signal
import time
from subprocess import Popen
import ntpath

class blah:

    def __init__(self):
        self.images = glob.glob('/home/james/.gazebo/models/pixl/materials/textures/*.png')
        self.subscriber = rospy.Subscriber('/pixl/spawn', String, self.spawn_callback)
        time.sleep(3)
        print("init done")

    def spawn_callback(self, msg):
        print("blah")
        n = random.randint(0,len(self.images))
        template = open('/home/james/.gazebo/models/pixl/materials/scripts/pixl_template','rb').read()
        image = ntpath.split(self.images[n])
        print image
        image = ntpath.basename(image[1])
        pixl = template % {'texture': image}
        material = open('/home/james/.gazebo/models/pixl/materials/scripts/pixl.material', 'wb')
        material.write(pixl)
        material.close()
        self.node = Popen('rosrun gazebo_ros spawn_model -x %0.3f -y 0 -z 0 -sdf -file /home/james/.gazebo/models/pixl/model-1_4.sdf -model pixl' % random.gauss(1.0,0.05), shell=True, preexec_fn=os.setsid)
        time.sleep(3)


    def shutdown(self):
        os.killpg(self.node.pid,signal.SIGTERM)

def main():
    rospy.init_node('blah', anonymous=True)
    SP = blah()
    rospy.on_shutdown(SP.shutdown)
    rospy.spin()

if __name__ == '__main__':
    main()
