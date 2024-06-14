#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import multiprocessing

import time
from multicam_cfg import *
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def openCamera(port):
    global img_width, img_height
    cap = cv2.VideoCapture(port)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))  # 视频流格式
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    
    exposure = 50.0 
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0) # enable auto exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0) # disable auto exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure) # manually setting exposure
    
    cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)   # enable auto whitening
    # cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)   # disable auto whitening
    
    width = cap.get(3) 
    height = cap.get(4) 
    frame_rate = cap.get(5) 
    print('== camera port ' + str(port) + ' ==') 
    print('- camera info') 
    print('image width: ', width, '\n', 
            'image hight: ', height, '\n', 
            'frame rate (Hz): ', frame_rate)
    
    print('- ros topic ')
    rospy.init_node('camera_stream', anonymous=True)
    if port == port_id1:
        pub_img = rospy.Publisher(camleft_topic, Image, queue_size=1)
        print('name: ' + camleft_topic)
    else:
        pub_img = rospy.Publisher(camright_topic, Image, queue_size=1)
        print('name: ' + camright_topic)
        
    seq = 0
    bridgeC = CvBridge()
    print('camera initialization done!')

    while not rospy.is_shutdown():
        ret, frame = cap.read()

        if not ret:
            print("get camera " + str(port) + " frame is empty")
            break
        
        msg_img = bridgeC.cv2_to_imgmsg(frame)
        msg_img.header.frame_id = 'cam_holder'
        msg_img.header.seq = seq
        msg_img.header.stamp = rospy.get_rostime()
        pub_img.publish(msg_img)
        seq += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # multi process the port index should be an even number
    cam1 = multiprocessing.Process(target=openCamera, args=(port_id1,))
    cam1.start()

    cam2 = multiprocessing.Process(target=openCamera, args=(port_id2,))
    cam2.start()
    
    
    # cam2 = multiprocessing.Process(target=openCamera, args=(8,))
    # cam2.start()
    