#!/usr/bin/env python

'''
Author: Dianye Huang
Date: 2022-08-08 14:23:56
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-14 21:03:06
Description: 
    
    Eye in hand calibration GUI, the intrinsic params are obtained 
    by running the cam_intrinsic_cali.py script.
    
    Calib result:
    The relative pose between panda_link8 to to camera pose
    
    # - Right
    # CaliCompute --Calibration results:
    right_pp = np.array([
        [-0.00880652, -0.84430906,  0.53578416,  0.03034515],
        [ 0.99979944,  0.0022032,   0.01990529,  0.03943157],
        [-0.01798666,  0.535852 ,   0.84412032,  0.03500989],
        [ 0.        ,  0.     ,     0.      ,    1.        ]
    ])
    
    # - Left
    # CaliCompute --Calibration results:
    left_pp = np.array([
        [-0.02259819, -0.54134383,  0.84049758,  0.03675402],
        [ 0.99927704,  0.01347901,  0.03554874, -0.04080249],
        [-0.03057317,  0.84069327,  0.54064786,  0.03747147],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    
'''

# qt
import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QMainWindow, QApplication, QWidget
from PyQt5.QtWidgets import QPushButton, QLabel, QGraphicsView, QGraphicsScene, QLineEdit
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout

# ros
import rospy
from sensor_msgs.msg import Image

# others
import cv2
from cv_bridge import CvBridge
from eyeinhand_utils import CaliCompute, CaliSampler, CaliBoardDetector

class CalibGUI(QMainWindow):
    '''
    Description: 
        CalibGUI serves as a handeye calibration manager, that assembling
        the samples by processing messages from ros, and utilized the 
        handeyecalibration api in cv2 to get the calibration result. The 
        caliration matrix is then broadcasted to rviz for an intuitive
        verification.
    @ param : width{int}  --width of the GUI
    @ param : height{int} --height of the GUI
    '''
    def __init__(self, 
                group_name="panda_arm", 
                base_link="panda_link0", 
                ee_link="panda_link8", 
                cali_method='Tsai-Lenz',
                width=800, 
                height=800,
                detector:CaliBoardDetector=None,
                img_topic=None,
                ):
        super().__init__()
        # 1. ROS init
        rospy.init_node('gui_handeye_calibration_node', anonymous=True)
        self.rate = rospy.Rate(50)

        self.group_name = group_name
        self.base_link = base_link
        self.ee_link = ee_link
        self.cali_method = cali_method

        # 2. Calibration modules
        self.sampler = CaliSampler(group_name=group_name, 
                                    base_link=self.base_link, 
                                    endeffector_link=self.ee_link)
        self.compute = CaliCompute()

        # -- Aruco detector
        self.aruco_res = None
        self.img = None
        self.res_img = None
        self.aruco_detector = detector
        self.bridgeC = CvBridge()
        self.cimg_sub = rospy.Subscriber(img_topic, Image, 
                                        self.sub_img_cb)
        rospy.wait_for_message(img_topic, Image, timeout=3)

        # 3. GUI init
        # -- layout
        self.log_cnt = 0
        self.prompt_str = "Do you want to record the current joint and ee states, and aruco results?\n" + \
                        "- Press sample to confirm recording joint states\n" + \
                        "- Press Quit to exit the program"
        self.width = width
        self.height = height
        self.initUI()
        # -- timers
        self.aruco_timer = QTimer()
        self.aruco_timer.timeout.connect(self.show_aruco_res)  # connect signals and slot
        self.aruco_timer.start(30)
        self.robot_timer = QTimer()
        self.robot_timer.timeout.connect(self.show_robot_states)
        self.robot_timer.start(30) 


    def show_aruco_res(self):
        if not self.img is None:
            self.aruco_res, self.res_img = self.aruco_detector.detect_charuco(self.img)
            self.res_img = cv2.cvtColor(self.res_img, cv2.COLOR_RGB2BGR)
            
            y, x = self.res_img.shape[:2]
            frame = QImage(self.res_img, x, y, QImage.Format.Format_RGB888)
            self.pix = QPixmap.fromImage(frame)
            self.item=QGraphicsPixmapItem(self.pix)
            self.scene = QGraphicsScene()
            self.scene.addItem(self.item)
            self.imgshow.setScene(self.scene)

        if not self.aruco_res is None:
            astr = '<b>Pose (px, py, pz, qx, qy, qz, qw):</b>'
            self.sampler.ar_trans = self.aruco_res[:3].copy()
            self.sampler.ar_quat = self.aruco_res[3:].copy()
            for p in self.sampler.ar_trans: # trans
                p = round(p, 2)
                astr += '\t' + str(p)
            for p in self.sampler.ar_quat: # quat
                p = round(p, 3) 
                astr += '\t' + str(p)
            self.ainfo.setText(astr)
        
    def show_robot_states(self):
        # end-effector states
        estr = '<b>End-effector pose (px, py, pz, qx, qy, qz, qw):</b>'
        self.sampler.ee_trans, self.sampler.ee_quat = self.sampler.get_pose_from_tf_tree(child_frame=self.ee_link, 
                                                                                        parent_frame=self.base_link)
        for t in self.sampler.ee_trans:
            t = round(t, 3)
            estr += '\t' + str(t)
        for q in self.sampler.ee_quat:
            q = round(q, 3)
            estr += '\t' + str(q)
        self.einfo.setText(estr)

    def sub_img_cb(self, ros_img):
        self.img = self.bridgeC.imgmsg_to_cv2(ros_img)
    
    def on_click_sample(self):
        self.log_cnt += 1
        self.prompt.setText('Have recorded '+ str(self.log_cnt) + 
                            ' samples<aruco pose and ee pose>\n' + 
                            self.prompt_str)
        if not self.aruco_res is None:
            self.compute.push_back_sample_buffer(self.sampler.ee_trans, self.sampler.ee_quat, 
                                                self.sampler.ar_trans, self.sampler.ar_quat)
            print('ee: ', self.sampler.ee_trans, self.sampler.ee_quat)
            print('ar: ', self.sampler.ar_trans, self.sampler.ar_quat)

    def on_click_compute(self):
        # get results
        cali_mat = self.compute.compute(method=self.cali_method)
    
    def initUI(self):
        # setting init geometry of GUI
        self.resize(self.width, self.height)

        # init widgets
        # 1. menu
        # pass

        # -3 sample button
        self.btn_sample = QPushButton('Sample', self)
        self.btn_sample.clicked.connect(self.on_click_sample)
        # -3 compute button
        self.btn_compute = QPushButton('Compute', self)
        self.btn_compute.clicked.connect(self.on_click_compute)
        # -4 quit button
        self.btn_quit = QPushButton('Quit', self)
        self.btn_quit.clicked.connect(QApplication.instance().quit)

        # 3. labels
        self.prompt = QLabel(self.prompt_str)
        self.ainfo = QLabel('<b>Pose (px, py, pz, qx, qy, qz, qw):</b>')
        self.einfo = QLabel('<b>End-effector pose (px, py, pz, qx, qy, qz, qw):</b>')

        # 4. graphics
        self.imgshow = QGraphicsView()
        self.imgshow.setObjectName('realsense_raw_color_imgs')


        # add status bar
        self.statusBar().showMessage('Ready')

        # 3. setup layout
        self.hl_settings = QHBoxLayout()
        self.hl_settings.addSpacing(10)
        self.hl_settings.addStretch(1)

        self.hl_button = QHBoxLayout()
        self.hl_button.addSpacing(10)
        self.hl_button.addStretch(1)
        self.hl_button.addWidget(self.btn_sample)
        self.hl_button.addWidget(self.btn_compute)
        self.hl_button.addWidget(self.btn_quit)

        self.layout = QGridLayout()
        self.layout.setSpacing(5)
        self.layout.addWidget(self.imgshow, 1, 0)
        self.layout.addWidget(self.ainfo,  3, 0)
        self.layout.addWidget(self.einfo,  4, 0)
        self.layout.addWidget(self.prompt, 5, 0)
        self.layout.addLayout(self.hl_settings, 6, 0)
        self.layout.addLayout(self.hl_button, 7, 0)

        # setup Qwidget in mainWindow
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.setWindowTitle('Franka Handeye Calibration Log cmd GUI')
        self.show()


from multicam_cfg import *
import cv2.aruco as aruco

if  __name__ == '__main__':
    
    ARUCO_DICT = aruco.DICT_4X4_250
    SQUARES_VERTICALLY = 5
    SQUARES_HORIZONTALLY = 7
    SQUARE_LENGTH = 0.035
    MARKER_LENGTH = 0.026
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard_create(SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, 
                                        SQUARE_LENGTH, MARKER_LENGTH, dictionary)

    left_cbd = CaliBoardDetector(
        img_height=img_height,
        img_width=img_width,
        img_D=[-7.97898732e-02,  2.49530609e-01,  1.02409919e-03, -5.47208067e-05, -2.44939489e-01],
        img_K=[358.06222443, 0.0, 309.14361973, 0.0, 357.74953614, 186.11295799, 0.0, 0.0, 1.0],
        marker_length=0.1,
        caliboard=board
    )
    
    right_cbd = CaliBoardDetector(
        img_height=img_height,
        img_width=img_width,
        img_D=[-0.08539184, 0.28856155, 0.00045643, 0.0005938, -0.30752491],
        img_K=[358.63432955, 0.0, 331.22704577, 0.0, 358.63717784, 186.44587027, 0.0, 0.0, 1.0],
        marker_length=0.1,
        caliboard=board
    )
    
    app = QApplication([])
    gui = CalibGUI(detector=left_cbd, img_topic='camera_left')
    # gui = CalibGUI(detector=right_cbd, img_topic='camera_right')
    sys.exit(app.exec())
    





