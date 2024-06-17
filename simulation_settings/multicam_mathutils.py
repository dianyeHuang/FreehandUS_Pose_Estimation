'''
Author: Dianye dianye.huang@tum.de
Date: 2024-06-15 17:27:06
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-15 17:45:33
FilePath: /FreehandUS/simulation_settings/multicam_mathutils.py
Description: 

'''

import numpy as np
import tf.transformations as t
from math import exp, sqrt, cos, sin, acos, floor
import matplotlib.pyplot as plt

class Quaternion:   
    def __init__(self, *args):
        if len(args) == 1:    
            q_arr = args[0]         # q_arr:np.array (qx, qy, qz, qw)
            self.eta = q_arr[3]
            self.epsilon = q_arr[:3:]
        elif len(args) == 2:   
            self.eta = args[0]      # eta:float (qw)
            self.epsilon = args[1]  # epsilon:np.array (qx, qy, qz)

    def skrew_mat(self, vec):
        x, y, z = vec
        return np.array([[ 0, -z,  y],
                        [ z,  0, -x],
                        [-y,  x,  0]]) 

    def __str__(self):
        op = ['i ', 'j ', 'k']
        result = ''
        if self.eta < -1e-8 or self.eta > 1e-8:
            result += str(round(self.eta, 4)) + ' '
        else:
            result += '0.0 '
        for i in range(3):
            val = self.epsilon[i]
            if (val < -1e-8) or (val > 1e-8):
                result += str(round(val, 4)) + op[i]
            else:
                result += '0.0' + op[i]
        return result

    def __add__(self, q):
        real = self.eta + q.eta
        imag = self.epsilon + q.epsilon
        return Quaternion(real, imag)
    
    def __sub__(self, q):
        real = self.eta - q.eta
        imag = self.epsilon - q.epsilon
        return Quaternion(real, imag)

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            real = self.eta*q.eta-np.dot(self.epsilon, q.epsilon)
            imag = self.eta*q.epsilon + q.eta*self.epsilon + \
                np.dot(self.skrew_mat(self.epsilon),q.epsilon)
            return Quaternion(real, imag)
        if isinstance(q, float) or isinstance(q, int):
            return Quaternion(q*self.eta, q*self.epsilon)
    
    def copy(self):
        return Quaternion(self.eta, self.epsilon)

    def conj(self):
        return Quaternion(self.eta, -1.0*self.epsilon)
    
    def wxyz_vec(self):
        return np.vstack((np.array(self.eta),self.epsilon.reshape(3,1)))
    
    def xyzw_vec(self):
        return np.array(self.epsilon.tolist() + [self.eta])




class QuatProc:
    def __init__(self):
        self.ZERO_THRESHOLD = 1e-7    
    
    def quat_process(self, q_list):
        # initial and goal quaternion
        q_arr = np.array(q_list)
        self.Q0 = Quaternion(q_arr[0])
        self.Qg = Quaternion(q_arr[-1])
        
        # -- get eQ
        eQ = list()
        for q in q_arr:
            Qc = Quaternion(q) # current quaternion
            eQ_tmp = self.e_Q_metric(self.Qg, Qc)
            eQ.append(eQ_tmp)
        eQ_arr = np.array(eQ)
        # return eQ_arr[:-2:,:]
        return eQ_arr
    
    def e_Q_metric(self, q1:Quaternion, q2:Quaternion):
        return 2*self.logarithmic_map(q1*q2.conj())  # 2log(q1*q2.conj()) 
    
    def logarithmic_map(self, q:Quaternion):
        qe_norm = np.linalg.norm(q.epsilon)
        if qe_norm < self.ZERO_THRESHOLD:
            return np.array([0,0,0])
        return acos(q.eta)*q.epsilon/qe_norm
    
    def exponential_map(self, r:np.array):
        r_norm = np.linalg.norm(r)
        if r_norm < self.ZERO_THRESHOLD:
            return Quaternion(1, np.array([0, 0, 0]))
        return Quaternion(cos(r_norm), sin(r_norm)/r_norm*r)
    
    def correct_quat(self, quat_list):
        qlast = np.array(quat_list[0])
        for idx in range(1, len(quat_list)):
            if np.linalg.norm(qlast-
                np.array(quat_list[idx])) > 0.5:
                quat_list[idx] = [  -quat_list[idx][0], -quat_list[idx][1],
                                    -quat_list[idx][2], -quat_list[idx][3]]
            qlast =  np.array(quat_list[idx])
        return quat_list
    

if __name__ == '__main__':    
    # euler = np.array([-181.0, 0.0, 0.0])/180.0*np.pi
    # quat = t.quaternion_from_euler(*euler)
    # print('quat1: ', quat)
    
    # euler = np.array([178.0, 0.0, 0.0])/180.0*np.pi
    # quat = t.quaternion_from_euler(*euler)
    # print('quat2: ', quat)
    
    Qpc = QuatProc()
    Q0 = t.quaternion_from_euler(0.0, 0.0, 0.0)
    Qc = t.quaternion_from_euler(
            *np.array([0.0, 180.0, 180.0])/180.0*np.pi
        )
    qe = Qpc.e_Q_metric(Quaternion(Qc), Quaternion(Q0))
    print('qe: ', qe)

