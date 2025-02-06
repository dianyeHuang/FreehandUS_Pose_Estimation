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
of the US probe, and then record the images from the 2 external cameras and 
the pose of the camera_base.

The related Objects' names are:
- "Ground": define a position-agnostic environment
- "USProbTip": control its motion
- "CameraHolder": read its pose with respect to the world frame
- "Vision_left/middle/right": read their images

This virtual environment is built up to regress the pose of the CameraHolder
based on the streamed RGB images. The application of this work is to achieve
accurate estimation of the freehand US probe in a relatively structured 
environment, so that we could reconstruct a 3D volume without using expensive
EM or optical tracking devices.

To ground the simulation results, we have to develop a physical devices in a
"ping-pang" manner. Adapting some design params of the CameraHolder and cali
results to the virtual ones, and vice versa. 

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


import os
import cv2
import numpy as np
from tqdm import tqdm

from multicam_simutils import rand2pose_parser
import tf.transformations as t

from trackedimgs_gen_configs import *

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
        
        
        # create folder
        if not os.path.exists(FOLDER_PATH):
            skip_flag = False
            data_idx = 0
            os.makedirs(FOLDER_PATH)
            fpose = open(os.path.join(FOLDER_PATH, "camholder_poses.txt"), 'w')
            tpose = open(os.path.join(FOLDER_PATH, "ustip_poses.txt"), 'w')
            print("Folder created: ", FOLDER_PATH)
        else:
            skip_flag = True
            print("Folder already exists ...")
        
        iidd = None
        for _ in tqdm(range(NUM_DATA)):
            
            if skip_flag:
                break
            
            # generate new poses for the tip of the probe:
            randpose = rand2pose_parser(
                rand_vec=np.random.uniform(-1.0, 1.0, 6),
                ws_lim=WS_LIM, offset_xyz=OFFSET_XYZ,
                offset_alpha=OFFSET_ALP
            )      
            
            randquat = t.quaternion_from_euler(*randpose[3:])
            set_object_pose(clientID, USProbTip_hl, relative_handle=table_hl, 
                            pos=randpose[:3], ort=randquat, flag_quat=True)
                            # pos=randpose[:3], ort=randpose[3:], flag_quat=False)
            
            # -- stream images
            retval_imgl, img_left   = read_vision_sensor_img(clientID, camhl_list[0])
            retval_imgr, img_right  = read_vision_sensor_img(clientID, camhl_list[1])
            img_cam    = np.hstack((img_left, img_right))

            # -- stream poses
            retval_pose, campose = get_object_pose(clientID, camHolder_hl, 
                                relative_handle=table_hl, flag_quat=False)
            
            # -- save image and pose data pair
            if retval_imgl and retval_imgr and retval_pose:
                img_svpath = os.path.join(FOLDER_PATH, "camholderimg" + 
                                          str(data_idx) + ".png")
                img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_svpath, img_cam)
                fpose.write(
                    str(list(campose[0])+list(campose[1]))[1:-1] + '\n'
                )
                tpose.write(
                    str(list(randpose))[1:-1] + '\n'
                )
                data_idx += 1
            
        ###########################################          
            
        # Close the connection to CoppeliaSim:
        if not skip_flag: 
            fpose.close()
            tpose.close()
        sim.simxFinish(clientID)
    else:
        print ('Failed connecting to remote API server')
    print ('Program ended')