#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-18 11:39:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-09-06 
Description: 
    Loading data for training
'''

import torch
from torch.utils.data import Dataset

import cv2
import os
import glob
import numpy as np
from PIL import Image
from natsort import natsorted

from pprint import pprint

# should be aligned with the WS_LIM definition in multicam_simmain.py
from trackedimgs_gen_configs import WS_LIM, OFFSET_XYZ, OFFSET_ALP
LABEL_LIM_NORM = np.array([
                WS_LIM[0]/2.0, WS_LIM[1]/2.0, WS_LIM[2]/2.0,
                WS_LIM[3]/2.0/180.0*np.pi, 
                WS_LIM[4]/2.0/180.0*np.pi,
                WS_LIM[5]/2.0/180.0*np.pi
            ])
LABEL_BALANCE = np.array([ # 1 deg -> 0.01745rad, 1 mm  -> 0.001 m
    15.0, 15.0, 15.0, 1.0, 1.0, 1.0
])

def predict_postproc(pred:np.array, 
                     multi_out=False, 
                     multi_n6=False, # true: batch x num_len x 6, false: batch x 6 x num_len
                     offset=False, 
                     normal=False, 
                     balance=False):
    global OFFSET_XYZ, WS_LIM, OFFSET_ALP, LABEL_LIM_NORM, LABEL_BALANCE
    if normal:
        assert offset == True, 'Offset should be true if normal is true!'

    if multi_out:
        if not multi_n6:
            pred = pred.transpose(0, 2, 1)
        pred = pred.reshape(-1, 6)
    
    if normal:
        pred *= LABEL_LIM_NORM
    elif balance:
        pred /= LABEL_BALANCE
    
    if offset:
        if len(pred.shape) == 2:
            pred[:, 0] += OFFSET_XYZ[0]
            pred[:, 1] += OFFSET_XYZ[1]
            pred[:, 2] += OFFSET_XYZ[2]+WS_LIM[2]/2.0
            pred[:, 3] += OFFSET_ALP
        else:
            pred[:4] += np.array([
                        OFFSET_XYZ[0],
                        OFFSET_XYZ[1],
                        OFFSET_XYZ[2]+WS_LIM[2]/2.0,
                        OFFSET_ALP,
                    ])
    return pred

def quaternion_dot_product(q1, q2):
    dot_product = np.dot(q1, q2)
    return dot_product

def quaternion_angle(q1, q2):
    dot_product = quaternion_dot_product(q1, q2)
    angle_rad = 2 * np.arccos(np.abs(dot_product))
    return angle_rad

def score_fn(pred, target, multi_out=False, offset=False, normal=False, balance=False):
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    if multi_out: target = target.reshape(-1, 6)
    pred = predict_postproc(pred.copy(), multi_out, offset, normal, balance)
    return np.mean(np.linalg.norm(pred[:, 3:]-target[:, 3:], axis=1)) +\
        10.0 * np.mean(np.linalg.norm(pred[:, :3]-target[:, :3], axis=1)) # 0.01 can guarantee good performance

class MultiCamDataset(Dataset):
    def __init__(self, folder_path, 
                 label_txt="ustip_poses.txt",
                 transform=None,
                 offset=False,
                 normal=False,
                 balance=False,
                 imgresize=None,
                 vstack=False,
                 gray=False,
                 image_format='*.png',
                 pin_image=False
                ):
        super().__init__()
        self.transform = transform
        self.vstack = vstack
        self.gray = gray
        self.pin_image = pin_image
        
        self.img_orgsize = None
        self.image_list  = list()
        self.label_list  = list()
        self.pose_list   = list()
        self.retpose     = False
        self.imgresize   = imgresize
        
        if normal:
            assert offset == True, 'Offset should be true if normal is true!'
        
        txt_path = os.path.join(folder_path, label_txt)
        os.chdir(folder_path)
        img_names = glob.glob(image_format)
        assert len(img_names) != 0, 'Check the image format ...' 
        img_names = natsorted(img_names)
        
        with open(txt_path) as f:
            for pose_str, img_name in zip(f.readlines(), img_names):
                # pose
                pose = list(map(float, pose_str.split(', ')))
                self.pose_list.append(np.array(pose).copy())
                
                if offset:
                    pose[0] -= OFFSET_XYZ[0]
                    pose[1] -= OFFSET_XYZ[1]
                    pose[2] -= OFFSET_XYZ[2]+WS_LIM[2]/2.0
                    pose[3] -= OFFSET_ALP
                label_pose = np.array(pose).copy() # mm & rad with offset
                
                if normal:
                    label_pose /= LABEL_LIM_NORM
                elif balance:
                    label_pose *= LABEL_BALANCE
                
                # image 
                img_path = os.path.join(folder_path, img_name)
                if self.img_orgsize is None:
                    tmp_img = Image.open(img_path)
                    self.img_orgsize = [tmp_img.size[1], tmp_img.size[0]]
                # append info
                if self.pin_image:
                    img = self._preproc_img(image_path=img_path)
                    self.image_list.append(img)
                else:
                    self.image_list.append(img_path)
                        
                self.label_list.append(label_pose)
    
    def set_retpose(self, retpose=True):
        self.retpose = retpose
       
    def __len__(self):
        return len(self.image_list)
    
    def _preproc_img(self, image_path):
        image = Image.open(image_path)
        if self.imgresize is not None:
            if self.vstack:
                image = np.asarray(image)
                image = np.vstack((
                    image[:, :self.img_orgsize[1]//2, :],
                    image[:, self.img_orgsize[1]//2:, :],
                ))
                tmp_image = cv2.resize(image, [self.imgresize[1], self.imgresize[0]])
                image = np.zeros_like(tmp_image)
                image[tmp_image>1] = 255.0
                
                if self.gray:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                tmp_image = image.resize([self.imgresize[1], self.imgresize[0]])
                image = np.zeros_like(tmp_image)
                image[tmp_image>1] = 255.0
        if self.transform:
            image = self.transform(image)
        return image
    
    def __getitem__(self, index):
        if self.pin_image:
            image = self.image_list[index]
        else:
            image = self._preproc_img(self.image_list[index])
        
        label = self.label_list[index]
        label = torch.Tensor(np.array(label))
        if self.retpose:
            pose = torch.Tensor(np.array(
                    self.pose_list[index]
                ))
            return image, label, pose
        return image, label
    
    def disp_info(self):
        print('---------- Dataset Info. ---------')
        print('Length of the dataset is:', len(self.image_list))
        image = Image.open(self.image_list[0])
        label = self.label_list[0]
        # print('type of image:', type(image))
        print('Original size of the image:', image.size)
        print('Original size of the label:', label.size)
        print('----------------------------------')
        
if __name__ == '__main__':
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    network_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(
        network_dir, 'dataset', 'example_1k_seg_color'
    )
    
    dataset = MultiCamDataset(
                    dataset_dir, 
                    label_txt="ustip_poses.txt",
                    transform=transforms.Compose([
                        transforms.ToTensor(), # pixel values is {0.0, 1.0}
                    ]),
                    offset=True,
                    normal=False,
                    balance=True,
                    imgresize=(256, 256),
                    vstack=True,
                    gray=False,
                    image_format='*.png'
                )
    dataset.set_retpose(True)
    train_loader = DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=4,
                                pin_memory=False)
    for images, labels, poses in train_loader:
        print('images shape: ', images.size())
        print('labels shape: ', labels.size())
        print('poses  shape: ', poses.size())
        break
    
    # import cv2
    # for i in range(img.shape[2]//2):
    #     cv2.imshow('hello check!', img[:,:, i*3:(i+1)*3])
    #     cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    
    # print(len(img))
    # print(label)
    
    
    