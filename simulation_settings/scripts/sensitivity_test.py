'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-14 21:38:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-17 19:10:10
FilePath: 
Description: 
1. Verify the sensitivity of the image changes when the probe moves
2. Collect images at the same time computing the overlap ratio (dice score)

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
import datetime

from multicam_simutils import rand2pose_parser
import tf.transformations as t


WS_LIM = [0.18, 0.15, 0.04, 40.0, 40.0, 90.0] # tx, tx, tz, eulerx, eulery, eulerz
OFFSET_XYZ = [-0.09, 0.0, 0.05] # offset x, y , z
OFFSET_ALP = np.pi

NUM_POSES  = int(100)
DOF_CHANGE = [3.0, 3.0, 3.0, 1.5, 1.5, 1.5] # mm & deg
DOF_CHANGE_list = np.hstack((np.array(DOF_CHANGE[:3])*1e-3,
                        np.array(DOF_CHANGE[3:])*np.pi/180.0))
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
FOLDER_PATH = 'xxx'


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_random_poses(num, savefolder=None):
    # generate new poses for the tip of the probe:
    pose_list = list()
    if savefolder is not None:
        savepath = os.path.join(savefolder, 'randposes.txt')
        rndpose = open(savepath, 'w') # 7x1 translation and euler (m and rad)
        rndpose.write(
                str(list(DOF_CHANGE_list))[1:-1] + '\n'
        )
        print('random poses save path: ', savepath)
    for _ in range(int(num)):
        randpose = rand2pose_parser(
            rand_vec=np.random.uniform(-1.0, 1.0, 6),
            ws_lim=WS_LIM, offset_xyz=OFFSET_XYZ,
            offset_alpha=OFFSET_ALP
        )      
        pose_list.append(randpose)
        if savefolder is not None:
            rndpose.write(
                str(list(randpose))[1:-1] + '\n'
            )
    if savefolder is not None:
        rndpose.close()
    return pose_list


def get_image_from_pose(clientID, USProbTip_hl, relative_handle, pose, camhl_list, flag_vis=False):
    
    quat = t.quaternion_from_euler(*pose[3:])
    set_object_pose(clientID, USProbTip_hl, relative_handle=relative_handle, 
                    pos=pose[:3], ort=quat, flag_quat=True)
    
    retval_imgl, img_left   = read_vision_sensor_img(clientID, camhl_list[0])
    retval_imgr, img_right  = read_vision_sensor_img(clientID, camhl_list[1])
    img_cam    = np.hstack((img_left, img_right))
    
    if retval_imgl and retval_imgr:
        img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)
        if flag_vis:
            cv2.imshow('dual cam images', img_cam)
            cv2.waitKey(33)
        return img_cam
    
    return None

from tqdm import tqdm
def collect_imageset(num_poses=NUM_POSES, flag_vis=False):
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
        
        ################ coding start here ##############
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
            create_folder_if_not_exists(FOLDER_PATH)
        else:
            skip_flag = True
            print("Folder already exists ...")
        
        if not skip_flag:
            pose_list = get_random_poses(num_poses, FOLDER_PATH)
            
            for pidx, pose in enumerate(tqdm(pose_list)):
                ppidx = 0
                camimg = get_image_from_pose(clientID, USProbTip_hl, table_hl, pose, camhl_list, flag_vis=flag_vis)
                if camimg is not None:
                    img_folder = os.path.join(FOLDER_PATH, str(pidx))
                    create_folder_if_not_exists(img_folder)
                    img_svpath = os.path.join(img_folder, str(ppidx)+'.jpg')
                    cv2.imwrite(img_svpath, camimg)
                ppidx += 1
                
                for d in range(6):
                    for c in [-1.0, 1.0]:
                        pose_c = pose.copy()
                        pose_c[d] = pose_c[d] + c*DOF_CHANGE_list[d]
                        
                        camimg = get_image_from_pose(clientID, USProbTip_hl, table_hl, pose_c, camhl_list, flag_vis=flag_vis)
                        if camimg is not None:
                            img_svpath = os.path.join(img_folder, str(ppidx)+'.jpg')
                            cv2.imwrite(img_svpath, camimg)
                        ppidx += 1
                        
        ################# coding end here ###########
        sim.simxFinish(clientID)
    else:
        print ('Failed connecting to remote API server')
    print ('Program ended')


def compute_IoU(bimg0, bimg1):
    # IoU
    sumimg = bimg0 + bimg1
    intersect = np.sum((sumimg>1.5).reshape(-1))
    union = np.sum((sumimg>0.5).reshape(-1))
    return intersect/union

def compute_dice(bimg0, bimg1):    
    # Dice
    union = np.sum((bimg0>0.5).reshape(-1)) + \
            np.sum((bimg1>0.5).reshape(-1))
    intersect = np.sum(((bimg0>0.5) & (bimg1>0.5)).reshape(-1))
    return 2*intersect/union

from natsort import natsorted
def analyze_dataset():
    file_names = natsorted(os.listdir(FOLDER_PATH))
    
    pose_change_list = list()
    for name in tqdm(file_names):
        folder_path = os.path.join(FOLDER_PATH, name)
        if os.path.isfile(folder_path): continue
        
        binimg_list = list()
        for j in range(13):
            img_path = os.path.join(folder_path, str(j)+'.jpg')
            img = cv2.imread(img_path) # original image
            _, bin_img0 = cv2.threshold(img[:, :, 0], 50, 255,cv2.THRESH_BINARY)  # res = 255 or 0
            _, bin_img1 = cv2.threshold(img[:, :, 1], 50, 255,cv2.THRESH_BINARY)
            _, bin_img2 = cv2.threshold(img[:, :, 2], 50, 255,cv2.THRESH_BINARY)
            bin_img = bin_img0 + bin_img1 + bin_img2

            binimg_list.append(bin_img.copy()/255.0)
            
        # compute IoU
        iou_list  = list()
        dice_list = list()
        for d in range(6):
            # IoU
            iou0 = compute_IoU(binimg_list[0], binimg_list[d*2])
            iou1 = compute_IoU(binimg_list[0], binimg_list[d*2+1])
            iou = (iou0 + iou1)/2.0
            # print(d, '-th IoU: ', iou)
            iou_list.append(iou)
            
            # Dice
            dice0 = compute_dice(binimg_list[0], binimg_list[d*2])
            dice1 = compute_dice(binimg_list[0], binimg_list[d*2+1])
            dice = (dice0 + dice1)/2.0
            # print(d, '-th Dice: ', dice)
            dice_list.append(dice)
        
        pose_change_list.append([iou_list, dice_list])
    
    
    '''
        Save results into a text file
    '''
    pose_change_arr = np.array(pose_change_list)
    print('pose: ', pose_change_arr.shape)

    mean_IoU = np.mean(pose_change_arr[:, 0, :], axis=0)
    print('mean_IoU: ', mean_IoU)
    
    mean_dice = np.mean(pose_change_arr[:, 1, :], axis=0)
    print('mean_dice: ', mean_dice)
    
    respath = os.path.join(FOLDER_PATH, 'result.txt')
    fopen = open(respath, 'w')
    DOF_CHANGE_str = str(list(DOF_CHANGE))[1:-1]
    mean_IoU_str  = str(list(mean_IoU))[1:-1]
    mean_dice_str = str(list(mean_dice))[1:-1]
    
    fopen.write(
        f'The change of each dimension is {DOF_CHANGE_str} in mm and deg; ' +
        f'The mean IoU of each dimension is {mean_IoU_str}; ' + 
        f'The mean Dice of each dimension is {mean_dice_str}. '
    )
    
    import matplotlib.pyplot as plt
    
    res_list = list()
    for res in pose_change_arr[:, 1, :]:
        if len(res_list) > 80:
            break    
        save_flag = True
        for val in res:
            if np.isnan(val): 
                save_flag = False
                break
        if save_flag: res_list.append(res)
    res_arr = np.array(res_list)
    
    # plt.plot(pose_change_arr[:, 1, :])
    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(res_arr)
    plt.xlabel('Random poses')
    plt.ylabel('Dice score')
    plt.legend(['tx', 'ty', 'tz', 'rx', 'ry', 'rz'], ncol=3)
    plt.savefig('xx/fig-sensitivity.png')
    plt.savefig('xx/fig-sensitivity.svg')
    # plt.figure('boxplot results')
    # plt.subplot(121)
    # plt.boxplot(trans_err)
    # plt.xlabel('Translation error (mm)')
    # plt.grid(True)

    # plt.subplot(122)
    # plt.boxplot(quat_err)
    # plt.xlabel('Angular error (deg)')
    # plt.grid(True)analyze_dataset

    plt.show()
    
    
    
if __name__ == '__main__':
    # collect_imageset(num_poses=NUM_POSES)
    analyze_dataset()