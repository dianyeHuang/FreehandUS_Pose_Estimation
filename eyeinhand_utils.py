#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-12 17:01:51
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-14 20:46:51
Description: 
    Some utility functions for eye in hand calibration. 
    CaliSampler, sample the pose of the end-effector and the pose of the marker (charuco in this repository.
    CaliCompute, compute the extrinsic params according to the sample pairs
    CaliBoardDetector, detect the pose of the charuco board. opencv version: 4.5.5 
'''


import cv2
import numpy as np
import tf.transformations as t

import rospy
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
import tf

class CaliSampler():
    def __init__(self, 
                group_name="panda_arm", 
                base_link="panda_link0",
                endeffector_link="panda_link8", 
                rosnode=False):
        
        if rosnode:
            rospy.init_node('sample_node', anonymous=False)
        
        # wait until tf tree is set up
        print('CaliSampler: waiting_for_message: move_group/status')
        rospy.wait_for_message('move_group/status', GoalStatusArray)
        print('CaliSampler: move_group/status okay!')
        
        # use moveit to position the robot 
        self.group_name = group_name
        self.base_link = base_link
        self.ee_link = endeffector_link
        self.max_vel_limit = 0.1
        self.commander = MoveGroupCommander(self.group_name)
        self.commander.set_max_velocity_scaling_factor(self.max_vel_limit)

        # one sample to be recorded 
        self.jpos = list()
        self.ee_trans = list()  # w.r.t base frame, in the Frana case is "panda_link0"
        self.ee_quat = list()   
        self.ar_trans = list()  # w.r.t camera frame, in the realsense case is "camera_color_optical_frame"
        self.ar_quat = list()

        # get ee pose from tf tree
        self.tf_listener = tf.TransformListener()
    
    def get_pose_from_tf_tree(self, 
                            child_frame='panda_link8', 
                            parent_frame='panda_link0'):
        '''
        Description: 
            get translation and quaternion of child frame w.r.t. the parent frame
        @ param : child_frame{string} 
        @ param : parent_frame{string} 
        @ return: trans{list} -- x, y, z
        @ return: quat{list}: -- x, y, z, w
        '''    
        self.tf_listener.waitForTransform(parent_frame, 
                                        child_frame, 
                                        rospy.Time(), 
                                        rospy.Duration(1.0))
        (trans,quat) = self.tf_listener.lookupTransform(parent_frame,  # parent frame
                                                        child_frame,   # child frame
                                                        rospy.Time(0))
        return trans, quat


class CaliCompute():
    '''
    Description: 
        The CaliCompute would load the sample from a file with predefined formatm or 
        obtains the sample during the acquisition, and then computes the calibration
        matrix, i.e. the homogeneous transformation matrix of the TCP coordinate 
        w.r.t. the base frame. [Eye in hand calibration program to be genralized]
    URLs for reference:
        https://blog.csdn.net/weixin_43994752/article/details/123781759
    @ param : {}: 
    @ return: {}: 
    param {*} self
    '''    
    def __init__(self):
        self.AVAILABLE_ALGORITHMS = {
            'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        # sample buffer for calibration purpose, each elements in the list are of np.array type
        self.ee2base_trans = list()  # end-effector to base 3x1 translation vector
        self.ee2base_rotmat = list() # end-effector to base 3x3 rotation matrix 
        self.ar2cam_trans = list()   # aruco to camera 3x1 translation vector
        self.ar2cam_rotmat = list()  # aruco to camera 3x3 rotation matrix

    def load_sample_buffer_from_file(self, filepath:str):
        with open(filepath, 'r') as f:
            txt_list = f.readlines()[1::]
            sample_list = []
            for txt_line in txt_list:
                txt_line = txt_line[0:-1]
                sample_line = [float(entry) for entry in txt_line.split('\t')]
                sample_list.append(sample_line)
        for s in np.array(sample_list):
            self.push_back_sample_buffer(s[7:10], s[10:14], s[14:17], s[17:21])

    def reset_sample_buffer(self):
        '''
        Description: 
            clear the sample buffer
        '''        
        self.ee2base_trans = list()  
        self.ee2base_rotmat = list() 
        self.ar2cam_trans = list()   
        self.ar2cam_rotmat = list() 

    def push_back_sample_buffer(self, ee2base_trans:list, 
                                    ee2base_quat:list, 
                                    ar2cam_trans:list,
                                    ar2cam_quat:list):
        '''
        Description: 
            get the translation (position) of the end-effector and aruco,
            and their orientation (quaternion) w.r.t. the base frame and
            the camera frame respectively.
        @ param : ee2base_trans{list} -- 3x1 end-effector position w.r.t. base frame
        @ param : ee2base_quat{list}  -- 4x1 end-effector orientation w.r.t. base frame
        @ param : ar2cam_trans{list}  -- 3x1 aruco position w.r.t. camera frame
        @ param : ar2cam_quat{list}   -- 4x1 aruco orientation w.r.t. camera frame
        '''
        self.ee2base_trans.append(np.array(ee2base_trans).reshape(3,1))         
        self.ee2base_rotmat.append(t.quaternion_matrix(ee2base_quat)[:3, :3])
        self.ar2cam_trans.append(np.array(ar2cam_trans).reshape(3,1))
        self.ar2cam_rotmat.append(t.quaternion_matrix(ar2cam_quat)[:3, :3])

    def compute(self, method='Tsai-Lenz', filepath=None):
        '''
        Description: 
            Computet the handeye calibration matrix from the sample buffer, with the help of cv2
            the default methods are 'Tsai-Lenz'
        @ param : method{string} -- some built-in method in opencv2 
        @ return: {}: 
        '''
        calires = np.identity(4)
        if not filepath is None:
            self.load_sample_buffer_from_file(filepath) 
        if len(self.ee2base_trans) > 3:
            cam2ee_rotmat, cam2ee_trans = cv2.calibrateHandEye(self.ee2base_rotmat, self.ee2base_trans,
                                                                    self.ar2cam_rotmat, self.ar2cam_trans,
                                                                    method=self.AVAILABLE_ALGORITHMS[method])
            calires[:3, :3] = cam2ee_rotmat
            calires[:3, 3] = cam2ee_trans.reshape(-1)
            print('CaliCompute --Calibration results:')
            print(calires)
        else:
            print('Sampling more before computing the calibration result ...')
        return calires
    
    
import rospy 
from sensor_msgs.msg import Image
import cv2.aruco as aruco
from cv_bridge import CvBridge


class CaliBoardDetector:
    def __init__(self, img_height, img_width, img_D, img_K, marker_length, caliboard):
        
        self.img_h = img_height
        self.img_w = img_width
        self.iner_mtx = np.array(img_K).reshape(3, 3)
        self.dist = np.array(img_D)
        self.marker_length = marker_length
        self.cvbridge = CvBridge()
        self.caliboard = caliboard

    def detect_charuco(self, cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        arucoParams = cv2.aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=arucoParams)  # First, detect markers
        aruco.refineDetectedMarkers(gray, self.caliboard, corners, ids, rejectedImgPoints)
        if ids is not None: # if there is at least one marker detected
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, self.caliboard)    
            im_with_charuco_board = aruco.drawDetectedCornersCharuco(cv2_img, charucoCorners, charucoIds, (0, 255, 0))
            
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, self.caliboard, 
                                                                self.iner_mtx, self.dist, 
                                                                np.empty(1), np.empty(1))  # posture estimation from a charuco board
            if retval == True:
                im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, self.iner_mtx, self.dist, rvec, tvec, 0.08)  # axis length 100 can be changed according to your requirement
            
            if len(rvec.shape) == 2:
                quat = self.rvec2quat(rvec.reshape(-1).astype(np.float64))
                return list(tvec.reshape(-1)) + list(quat), im_with_charuco_board

        return None, cv2_img
    
    def rvec2quat(self, rvec):
        htfmtx = np.array([[0, 0, 0, 0], # homogenous transformation matrix
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1]],
                        dtype=float)
        htfmtx[:3, :3], _ = cv2.Rodrigues(rvec)
        quat = t.quaternion_from_matrix(htfmtx)
        return quat  # [qx, qy, qz, qw]
    

    def detect_aruco(self, cv2_img):
        
        aruco_detect_res = dict()
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        center_corners = np.array([])
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, rejectedImgPoints = aruco.estimatePoseSingleMarkers(corners, self.marker_length, 
                                                                            (self.iner_mtx), (self.dist))
            np_corners = np.array(corners)
            for idx, id in enumerate(ids):
                tmp_corners = np_corners[idx]
                center_corners = tmp_corners.mean(axis=1)
                center_corners = np.rint(center_corners)
                aruco_tmp_res = dict()
                aruco_tmp_res['rvec'] = rvec[idx][0]
                aruco_tmp_res['tvec'] = tvec[idx][0]
                aruco_tmp_res['corners'] = tmp_corners
                aruco_tmp_res['ccorner'] = center_corners
                aruco_detect_res[str(id[0])] = aruco_tmp_res

            for i in range(0, ids.size):
                aruco.drawAxis(cv2_img, self.iner_mtx, self.dist, rvec[i], 
                                tvec[i], self.marker_length/2)
            aruco.drawDetectedMarkers(cv2_img, corners)
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '
            cv2.putText(cv2_img, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(cv2_img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        return cv2_img, aruco_detect_res

    def img_cb(self, msg:Image):
        cv2_img = self.cvbridge.imgmsg_to_cv2(msg)
        # cv2_img, res = self.detect_aruco(cv2_img) # depends on detect charuco or aruco
        res, cv2_img = self.detect_charuco(cv2_img)
        
        print('res(t, q): ', res)
        
        # cv2.imshow('hello', aa.astype(np.uint8))
        cv2.imshow('hello', cv2_img)
        cv2.waitKey(1)
    
