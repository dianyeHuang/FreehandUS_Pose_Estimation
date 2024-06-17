'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-17 14:43:46
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-17 17:58:55
FilePath: /FreehandUS/simulation_settings/tmp.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np

# workspace_shrink = 0.8
# z_offset = 0.03 # 3.0 cm offset
# alpha_offset = np.pi
# lim_arr = np.array([
#     workspace_shrink*np.array([-1.0, 1.0])*0.35,  # pos x (m)
#     workspace_shrink*np.array([-1.0, 1.0])*0.25,  # pos y
#     np.array([ 0.0, 1.0])*0.08,                   # pos z
#     np.pi/180.0*np.array([-1.0, 1.0])*45.0,  # ort alpha (deg)
#     np.pi/180.0*np.array([-1.0, 1.0])*45.0,  # ort beta 
#     np.pi/180.0*np.array([-1.0, 1.0])*100.0  # ort gamma
# ])

# print('lim_arr: ', lim_arr)
from multicam_simutils import rand2pose_parser
randvec = np.random.uniform(-1.0, 1.0, 6)

pose = rand2pose_parser(
    rand_vec=randvec,
    ws_lim=[0.25, 0.20, 0.05, 60.0, 60.0, 200.0],
    offset_xyz=[-0.125, 0.0, 0.03],
    offset_alpha=np.pi
)
print('rand vector shape:', randvec.shape)
print('rand vector value:', randvec)
print('pose shape: ', pose.shape)
print('pose value: ', pose)