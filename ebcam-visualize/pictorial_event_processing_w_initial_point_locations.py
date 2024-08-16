#!/usr/bin/env python3

import rospy
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse

from rospy.numpy_msg import numpy_msg
from evetac_software.msg import EBFloats
from evetac_software.msg import EBInt8
from evetac_software.msg import EBInt16
from std_msgs.msg import Int64
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
import copy
from evetac_software.msg import EvetacMsg


class EventProcessor:
    def __init__(self, cam_id):

        rospy.Subscriber("/evetac_msg" + cam_id, EvetacMsg, self.callback_events)
        self.center_locations = None
        self.initial_center_locations = None
        self.br = CvBridge()
        self.interm_image = np.zeros((640,480),dtype=np.int8)
        self.counter = 0

        self.pub_num_events = rospy.Publisher('intimg' + cam_id, Image,queue_size=10)

    def callback_events(self, data):
        self.counter += 1
        # add events to the picture
        img_tmp = self.br.imgmsg_to_cv2(data.image, desired_encoding='passthrough')
        self.interm_image += np.clip(img_tmp,126,128)-127

        self.center_locations = np.fliplr(np.asarray(data.centers).reshape(-1,2, order='F').transpose())

        if (self.initial_center_locations is None):
            self.initial_center_locations = copy.deepcopy(self.center_locations)

        if (self.counter==1):
            self.counter = 0

            if not(self.center_locations is None):
                desired_locations = self.center_locations.astype(int)
                self.interm_image[(desired_locations[1,:], desired_locations[0,:])] = 100
                for i in range(10):
                    self.interm_image[(desired_locations[1,:]+i, desired_locations[0,:])] = 100
                    self.interm_image[(desired_locations[1,:], desired_locations[0,:]+i)] = 100

            if not(self.initial_center_locations is None):
                desired_locations = self.initial_center_locations.astype(int)
                self.interm_image[(desired_locations[1, :], desired_locations[0, :])] = -100
                for i in range(10):
                    self.interm_image[(desired_locations[1,:]+i, desired_locations[0,:])] = -100
                    self.interm_image[(desired_locations[1,:], desired_locations[0,:]+i)] = -100

            img = np.clip(50*self.interm_image + 128,0,255).astype(np.uint8)
            img = self.br.cv2_to_imgmsg(img,encoding="mono8")
            self.pub_num_events.publish(img)
            self.interm_image[:] = 0

def main():
    rospy.init_node('eventprocessor', anonymous=True)
    cam_id = rospy.get_param(rospy.get_name() + "/cam_id")
    EventProcessor(cam_id)
    rospy.spin()

if __name__ == '__main__':
    main()
