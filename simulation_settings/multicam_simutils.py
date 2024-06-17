'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-14 21:38:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-17 18:38:30
FilePath: /FreehandUS/simulation_settings/vrepsim_utils.py
Description: 
    when streaming the data, recall "simx_opmode_streaming" before using "simx_opmode_buffer"
    for example:
    _,peg_position =vrep.simxGetObjectPosition(clientID, peg_handle, -1, vrep.simx_opmode_streaming)
    _,peg_position = vrep.simxGetObjectPosition(clientID, peg_handle, -1, vrep.simx_opmode_buffer)
'''

import sim
import cv2
import numpy as np

def get_random_uniform(min_val, max_val):
    unirand = np.random.uniform(0.0, 1.0, len(min_val)).reshape(-1, 1)
    return min_val + unirand*(max_val-min_val)
    

def read_vision_sensor_img(clientID, sensor_handle, flag_color=True):
    """
        return an RGB image
    """
    # 1. read raw images
    option = 0 if flag_color else 1
    errCode, reso, raw_img = sim.simxGetVisionSensorImage(clientID, sensor_handle, 
                                                option, sim.simx_opmode_blocking)
    # 2. process raw image
    out_img = np.asarray(raw_img, dtype=np.float32)
    out_img[out_img < 0] += 255.0
    if flag_color:
        out_img.shape = (reso[1], reso[0], 3)
    else:
        out_img.shape = (reso[1], reso[0])
    out_img = np.flipud(out_img).astype(np.uint8)
    
    return out_img

def read_vision_sensor_depth(clientID, sensor_handle, zNear = 0.01, zFar = 2):
    """
        return a depth image
    """
    # 1. read raw data
    errCode, reso, depth_buffer = sim.simxGetVisionSensorDepthBuffer(clientID, 
                                        sensor_handle, sim.simx_opmode_blocking)
    # 2. process depth buffer
    depth_img = np.asarray(depth_buffer)
    depth_img.shape = (reso[1], reso[0])
    depth_img = np.flipud(depth_img)
    depth_img = depth_img * (zFar - zNear) + zNear
    
    return depth_img

def set_object_pose(clientID, obj_handle, pos=None, ort=None, relative_handle=-1, flag_quat=False):
    if pos is not None:
        sim.simxSetObjectPosition(clientID, obj_handle, relative_handle, 
                                            pos, sim.simx_opmode_blocking)
    if ort is not None:
        if flag_quat:
            sim.simxSetObjectQuaternion(clientID, obj_handle, relative_handle, 
                                                ort, sim.simx_opmode_blocking)
        else:
            sim.simxSetObjectOrientation(clientID, obj_handle, relative_handle,
                                                ort, sim.simx_opmode_blocking)

def get_object_pose(clientID, obj_handle, relative_handle=-1, flag_quat=False):
    errCode, pos = sim.simxGetObjectPosition(clientID, obj_handle,
                        relative_handle, sim.simx_opmode_blocking)
    if flag_quat:
        errCode, ort = sim.simxGetObjectQuaternion(clientID, obj_handle,
                            relative_handle, sim.simx_opmode_blocking)  # w, x, y, z
    else:
        errCode, ort = sim.simxGetObjectOrientation(clientID, obj_handle,
                            relative_handle, sim.simx_opmode_blocking)  # alpha, beta, gamma
    return pos, ort 

def rand2pose_parser(rand_vec, ws_lim, offset_xyz=[0.0, 0.0, 0.0], 
                    offset_alpha=np.pi, xy_shrink=1.0):
    return np.array([
        xy_shrink*rand_vec[0]*ws_lim[0]/2.0+offset_xyz[0],    # pos x (m)
        xy_shrink*rand_vec[1]*ws_lim[1]/2.0+offset_xyz[1],    # pos y
        (rand_vec[2]+1)*ws_lim[2]/2.0+offset_xyz[2],              # pos z
        np.pi/180.0*rand_vec[3]*ws_lim[3]/2 + offset_alpha,  # ort alpha (deg)
        np.pi/180.0*rand_vec[4]*ws_lim[4]/2,  # ort beta 
        np.pi/180.0*rand_vec[5]*ws_lim[5]/2   # ort gamma
    ])



if __name__ == '__main__':
    print('hello world!')

