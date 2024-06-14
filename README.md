<!--
 * @Author: Dianye dianye.huang@tum.de
 * @Date: 2024-06-14 20:28:11
 * @LastEditors: Dianye dianye.huang@tum.de
 * @LastEditTime: 2024-06-14 21:00:04
 * @FilePath: /FreehandUS/README.md
 * @Description: 
    * 
-->
# FreehandUS
Pose estimation of freedhand ultrasound probe under a semi-structured environment.

This repository includes:

- Dual camera images streaming
  - multicam_cfg.py
  - multicam_stream.py
  - multicam_test.py

- camera intrinsic and extrinsic parameters calibration, the calibration boards can be found in "assets/calibration_board". To generate the calibration board, we suggest: https://calib.io/pages/camera-calibration-pattern-generator .
  - cam_intrinsic_cali.py
  - eyeinhand_utils.py & eyeinhand_cali.py

- simulation images and pose pair collection, the simulation scene in coppeliaSim can be found in "assets/coppeliaSim_ttt" 
  - 

