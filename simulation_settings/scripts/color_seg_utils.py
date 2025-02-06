'''
    Segment the color markers based on color thresholds
'''

import cv2
import numpy as np

# https://blog.csdn.net/wanggsx918/article/details/23272669
# COLOR_HSV_RANGE = {
#     'red'  : [[np.array([156,  43, 46]),
#                np.array([  0,  43, 46])], 
#               [np.array([180, 255, 255]),
#                np.array([  5, 255, 255])] ],  # low , high    
#             #    np.array([ 10, 255, 255])] ],  # low , high    
#     # 'blue' : [np.array([100, 43, 46]), np.array([124, 255, 255])], # okay
#     'blue' : [np.array([100, 123, 46]), np.array([124, 255, 255])], 
#     # 'green': [np.array([ 35, 43, 46]), np.array([ 77, 255, 255])]  # not okay
#     'green': [np.array([ 35, 123, 100]), np.array([ 77, 255, 255])]  
# }
COLOR_HSV_RANGE = {
    'red'  : [[np.array([156,  43, 46]),
               np.array([  0,  43, 46])], 
              [np.array([180, 255, 255]),
               np.array([  5, 255, 255])] ],  # low , high    
            #    np.array([ 10, 255, 255])] ],  # low , high    
    # 'blue' : [np.array([100, 43, 46]), np.array([124, 255, 255])], # okay
    'blue' : [np.array([100, 123, 46]), np.array([124, 255, 255])], 
    # 'green': [np.array([ 35, 43, 46]), np.array([ 77, 255, 255])]  # not okay
    'green': [np.array([ 35, 120, 100]), np.array([ 77, 255, 255])]  
}


def segment_color_obj(hue_image, low_range, high_range):
    if len(low_range) == 2:
        th = cv2.add(
                cv2.inRange(hue_image, low_range[0], high_range[0]), 
                cv2.inRange(hue_image, low_range[1], high_range[1]))  
    else:
        th = cv2.inRange(hue_image, low_range, high_range)
    # cv2.imshow('th', th)
    # cv2.waitKey(3)
    
    dilated = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dilated = cv2.erode(dilated, None, iterations=2)
    bin_img = cv2.dilate(dilated, None, iterations=2)
    
    return bin_img

def find_centers(bin_img, max_num=2):
    (cnts, _) = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    closed_cs = sorted(cnts, key=cv2.contourArea, reverse=True)

    center_list = list()
    for c in closed_cs[:int(max_num)]:  # Limit to the first 2 largest contours
        # rect = cv2.minAreaRect(c)
        # box = np.intp(cv2.boxPoints(rect))
        # center = tuple(np.mean(np.array(box), axis=0).reshape(-1).astype(np.uint32))
        
        moment = cv2.moments(c)  
        center_x = int(moment["m10"] / moment["m00"])
        center_y = int(moment["m01"] / moment["m00"])
        center = [center_x, center_y]
        
        center_list.append(center)
        
    return center_list

def plot_cengters(img, centers, color=(0, 255, 0)):
    for center in centers:
        img = cv2.circle(img, center, 5, color, 2)
    return img

def color_seg(hue_image, get_centers=True, show=False, savepath=None):
    '''
        input image is the hue_image in HSV space,
        output is a color segmentation image, 
        with max value of 255 and min value of 0
    '''
    centers = list()
    rbin = segment_color_obj(hue_image, *COLOR_HSV_RANGE['red'])  # 0 or 255
    gbin = segment_color_obj(hue_image, *COLOR_HSV_RANGE['green'])
    bbin = segment_color_obj(hue_image, *COLOR_HSV_RANGE['blue'])
    seg_img = np.stack((rbin, gbin, bbin), axis=-1)  # height, width, channel. RGB
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    
    
    if show or get_centers or savepath is not None:
        # get_centers, label it into the images
        rcenters = find_centers(rbin, max_num=2)
        gcenters = find_centers(gbin, max_num=2)
        bcenters = find_centers(bbin, max_num=2)
        dis_image = plot_cengters(show.copy(), rcenters+gcenters+bcenters)
        
        catimg = np.vstack((dis_image, seg_img))
        if show:
            cv2.imshow('segmentation', catimg)
            cv2.waitKey(33)
        if savepath is not None:
            cv2.imwrite(savepath, catimg)
            
        if get_centers:
            centers = rcenters+gcenters+bcenters

    return seg_img, centers

if __name__ == '__main__':
    pass