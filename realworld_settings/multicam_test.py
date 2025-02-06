'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-07 17:05:25
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-14 16:33:04
Description:
    test streaming two cameras simultaneously
'''


import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
from multicam_cfg import *

class MultiCam_ImageCap:
    def __init__(self, img_height, img_width):
        rospy.init_node('camera_holder', anonymous=False)
        rospy.Subscriber(camleft_topic, Image, self.leftimg_cb)
        rospy.Subscriber(camright_topic, Image, self.rightimg_cb)
        
        self.img_width  = img_width
        self.img_height = img_height
        self.bridge = CvBridge()
        self.cam_image = np.zeros((img_height, img_width*2, 3), dtype=np.uint8)
    
    def leftimg_cb(self, msg:Image):
        img = self.bridge.imgmsg_to_cv2(msg)
        self.cam_image[:, :self.img_width, :] = img.copy()
    
    def rightimg_cb(self, msg:Image):
        img = self.bridge.imgmsg_to_cv2(msg)
        self.cam_image[:, self.img_width:self.img_width*2, :] = img.copy()
        

if __name__ == '__main__':
    mcic = MultiCam_ImageCap(img_height=img_height, 
                             img_width=img_width)
     
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # aa = cv2.resize(mcic.cam_image, ((img_width*2)//3, img_height//3))
        # aa = cv2.resize(mcic.cam_image, ((img_width*2)//2, img_height//2))
        aa = mcic.cam_image
        cv2.imshow('hello world', aa)
        cv2.waitKey(1)
        rate.sleep()
    


