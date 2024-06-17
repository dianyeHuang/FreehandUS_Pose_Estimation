'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-14 21:38:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-17 19:10:10
FilePath: /FreehandUS/simulation_settings/vrepsim_utils.py
Description: 
    
The dynamics engine is disabled for faster simulation.

Some useful links:
    - https://manual.coppeliarobotics.com/en/remoteApiFunctionsPython.htm

We setup a virtual environment where we first randomly control the movement
of the US probe, and then record the images from the 3 external cameras and 
the pose of the camera_base.

The related Objects' names are:
- "Ground": define a position-agnostic environment
- "USProbTip": control its motion
- "CameraHolder": read its pose with respect to the world frame
- "Vision_left/middle/right": read their images

This virtual environment is built up to regress the pose of the CameraHolder
based on the streamed RGB images. The application of this work is to achieve
accurate estimation of the freehand US probe in a relatively structured 
environment, so that we could reconstruct a 3D volume without using expasive
EM or optical tracking devices.


To ground the simulation results, we have to develop a physical devices in a
"ping-pang" manner. Adapting some design params of the CameraHolder and cali
results to the virtual ones, and vice versa. 


This script can collect two types of dataset, one is collected by randomly
placing the US probe within the workspace, the other is collected by sending
sequential movement commands to the virtual environment, which mimics the 
real application scenarios. The latter dataset enable the network to consider
consistency of the sequential output during the training.

Credit to the 3D models.


Segmentation anything:
https://github.com/facebookresearch/segment-anything/tree/main

'''


from multicam_simutils import (
    read_vision_sensor_img, get_object_pose, 
    set_object_pose, get_random_uniform
)

try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from multicam_simutils import rand2pose_parser

WS_LIM = [0.25, 0.20, 0.05, 60.0, 60.0, 200.0] # tx, tx, tz, eulerx, eulery, eulerz
OFFSET_XYZ = [-0.125, 0.0, 0.03] # offset x, y , z
OFFSET_ALP = np.pi


if __name__ == '__main__':
    """
        To save space, the images can be saved as *.jpg other than pickle file 
    """
    print ('Program started')
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5) # Connect to CoppeliaSim
    print('clientID: ', clientID)
    
    if clientID!=-1:
        print ('Successfully connected to remote API server ...')
        
        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        res,objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
        if res==sim.simx_return_ok:
            print ('Number of objects in the scene: ',len(objs))
        else:
            print ('Remote API function call returned with error code: ',res)
        
        ################ coding here ##############
        # - 1. Get handles
        errCode, camHolder_hl = sim.simxGetObjectHandle(clientID, 'CameraHolder' ,sim.simx_opmode_blocking)
        errCode, USProbTip_hl = sim.simxGetObjectHandle(clientID, 'USProbTip' ,sim.simx_opmode_blocking)
        errCode, table_hl = sim.simxGetObjectHandle(clientID, 'Table' ,sim.simx_opmode_blocking)
        
        errCode, left_hl   = sim.simxGetObjectHandle(clientID, 'Vision_left' ,sim.simx_opmode_blocking)
        errCode, right_hl  = sim.simxGetObjectHandle(clientID, 'Vision_right' ,sim.simx_opmode_blocking)
        camhl_list = [left_hl, right_hl]
        
        for i in range(200):
            # generate new poses for the tip of the probe:
            randpose = rand2pose_parser(
                rand_vec=np.random.uniform(-1.0, 1.0, 6),
                ws_lim=WS_LIM, offset_xyz=OFFSET_XYZ,
                offset_alpha=OFFSET_ALP
            )      
            set_object_pose(clientID, USProbTip_hl, relative_handle=table_hl, 
                            pos=randpose[:3], ort=randpose[3:], flag_quat=False)
            
            # -- stream images
            # img_left   = read_vision_sensor_img(clientID, camhl_list[0])
            # img_right  = read_vision_sensor_img(clientID, camhl_list[1])
            # img_cam    = np.hstack((img_left, img_right))
            # cv2.imwrite('abc.jpg', img_cam)
            
            # -- stream pose
            # cam_pos, cam_ang = get_object_pose(clientID, camHolder_hl, 
            #                     relative_handle=table_hl, flag_quat=False)
            # print('cam_pos: ', cam_pos)
            # print('cam_ang: ', cam_ang)
            
            ###########################################          
            
        # Now close the connection to CoppeliaSim:
        # cv2.destroyAllWindows()
        sim.simxFinish(clientID)
    else:
        print ('Failed connecting to remote API server')
    print ('Program ended')