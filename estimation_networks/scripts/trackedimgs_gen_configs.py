'''
    should be aligned with the workspace generated in the simulation
'''

import numpy as np

NUM_DATA = int(2e5)
FOLDER_PATH = 'xxxx'

WS_LIM = [0.18, 0.15, 0.04, 40.0, 40.0, 90.0] # tx, tx, tz, eulerx, eulery, eulerz
OFFSET_XYZ = [-WS_LIM[0]/2.0, 0.0, 0.05] # offset x, y , z
OFFSET_ALP = np.pi

if __name__ == '__main__':
    pass
