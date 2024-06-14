'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-07 15:57:43
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-14 20:37:42
Description: 
    configuration of the camera streaming, ros topic is utilized to publish the captured images
'''

import numpy as np

# camera port id
port_id1 = 6
port_id2 = 4

# image resolution
# possible image resolution includes
# 1920 x 1080
# 1280 x 720
# 800 x 600
# 640 x 480
# 640 x 360
# 320 x 240
# 320 x 180

# - 1080p  16:9
# img_width  = 1920
# img_height = 1080

# - 720p 16:9
# img_width  = 1280
# img_height = 720

# - 360p 16:9
img_width  = 640
img_height = 360

# rostopic settings
camleft_topic  = 'camera_left'
camright_topic = 'camera_right'

