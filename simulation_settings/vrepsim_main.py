'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-14 21:38:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-14 21:44:38
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


from vrepsim_utils import (
    read_vision_sensor_img, read_vision_sensor_depth,
    get_object_pose, set_object_pose, get_random_uniform
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

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    
    flag_train  = True # if false, the probe actions will be read from the training pickles
    flag_random = True
    group_idx   = 1
    
    # data directory
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    if flag_train:
        if flag_random:
            rel_dir = 'synthetic_data/train_data/random_pose/'
        else:
            rel_dir = 'synthetic_data/train_data/seq_pose/'
    else:
        if flag_random:
            rel_dir = 'synthetic_data/seg_data/random_pose/'
            data_actdir = os.path.join(pkg_dir, 
                            'synthetic_data/train_data/random_pose/')
        else:
            rel_dir = 'synthetic_data/seg_data/seq_pose/'
            data_actdir = os.path.join(pkg_dir, 
                            'synthetic_data/train_data/seq_pose/')
            
    data_savedir = os.path.join(pkg_dir, rel_dir)
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
        errCode, ground_hl = sim.simxGetObjectHandle(clientID, 'Ground' ,sim.simx_opmode_blocking)
        
        errCode, left_hl   = sim.simxGetObjectHandle(clientID, 'Vision_left' ,sim.simx_opmode_blocking)
        errCode, middle_hl = sim.simxGetObjectHandle(clientID, 'Vision_middle' ,sim.simx_opmode_blocking)
        errCode, right_hl  = sim.simxGetObjectHandle(clientID, 'Vision_right' ,sim.simx_opmode_blocking)
        camhl_list = [left_hl, middle_hl, right_hl]
        
        # - 2. Interacting with environment
        flag_vis   = True
        flag_seq   = False  # sequential movement or random placement
        num_sample = 1
        # workspace constraints        
        workspace_shink = 0.8
        lim_arr = np.array([
            workspace_shink*np.array([-1.0, 1.0])*0.35,  # pos x (m)
            workspace_shink*np.array([-1.0, 1.0])*0.25,  # pos y
            np.array([ 0.0, 1.0])*0.30,                  # pos z
            np.pi + np.pi/180.0*np.array([-1.0, 1.0])*45.0,  # ort alpha (deg)
            np.pi/180.0*np.array([-1.0, 1.0])*45.0,  # ort beta 
            np.pi/180.0*np.array([-1.0, 1.0])*180.0  # ort gamma
        ])
        
        if flag_seq:
            tip_pos, tip_ang = get_object_pose(clientID, USProbTip_hl, 
                                relative_handle=ground_hl, flag_quat=False)
            tip_pos, tip_ang = np.array(tip_pos), np.array(tip_ang)
        
        data_dict = dict()
        data_dict['res_imgs'] = None
        data_dict['cam_imgs'] = list()
        data_dict['cam_pose'] = list()
        data_dict['tip_pose'] = list()
        
        if flag_train:
            pbar = tqdm(range(int(num_sample)))
        else:
            with open(data_actdir+str(group_idx)+'.pkl', 'rb') as f:
                ret_dict = pickle.load(f)
            pbar = tqdm(ret_dict['tip_pose'])
            
        for act in pbar:
            # -- Set commands
            if flag_train:
                if flag_seq:
                    # actions w.r.t. the tip frame
                    pos_act = np.array([0.001, 0.002, 0.005])
                    ang_act = np.array([0.0, 0.0, 5.0])/180.0*np.pi
                    
                    # actions w.r.t the ground frame, constrain the movement of the US probe within a given space
                    tip_pos += pos_act
                    tip_ang += ang_act 
                else:
                    random_pose = get_random_uniform(lim_arr[:, 0].reshape(-1, 1), 
                                                    lim_arr[:, 1].reshape(-1, 1))
                    tip_pos   = random_pose[:3, :]
                    tip_ang = random_pose[3:, :]
            else:
                tip_pos = act[:3]
                tip_ang = act[3:]
            set_object_pose(clientID, USProbTip_hl, relative_handle=ground_hl, 
                                    pos=tip_pos, ort=tip_ang, flag_quat=False)
            
            # -- stream images
            flag_color = True
            img_left   = read_vision_sensor_img(clientID, camhl_list[0], flag_color)
            img_middle = read_vision_sensor_img(clientID, camhl_list[1], flag_color)
            img_right  = read_vision_sensor_img(clientID, camhl_list[2], flag_color)
            img_cam    = np.hstack((img_left, img_middle, img_right))
            
            cv2.imwrite('abc.jpg', img_cam)
            
            # -- stream pose
            cam_pos, cam_ang = get_object_pose(clientID, camHolder_hl, 
                                relative_handle=ground_hl, flag_quat=False)
            
            # -- data dict
            if data_dict['res_imgs'] is None:
                data_dict['res_imgs'] = img_left.shape
            data_dict['cam_imgs'].append(img_cam.copy())
            data_dict['cam_pose'].append([*cam_pos, *cam_ang])
            data_dict['tip_pose'].append([*tip_pos, *tip_ang])
                
            # --- image visualization
            if flag_vis:
                disp_img = cv2.cvtColor(img_cam, cv2.COLOR_RGB2BGR)
                cv2.imshow('read_vision_sensor', disp_img)
                cv2.waitKey(1)        
                # plt.figure()
                # plt.imshow(disp_img)
                # plt.show()
            
        # -- save data 1. images, 2. cam pose, 3. tip pose (position and euler angles)
        #    tip pose is the motion command, cam poses are output, images are input
        with open(data_savedir+str(group_idx)+'.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        ###########################################          
            
        # Now close the connection to CoppeliaSim:
        cv2.destroyAllWindows()
        sim.simxFinish(clientID)
    else:
        print ('Failed connecting to remote API server')
    print ('Program ended')