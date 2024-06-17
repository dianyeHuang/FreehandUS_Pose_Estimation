<!--
 * @Author: Dianye dianye.huang@tum.de
 * @Date: 2024-06-14 20:28:11
 * @LastEditors: Dianye dianye.huang@tum.de
 * @LastEditTime: 2024-06-15 17:17:44
 * @FilePath: /FreehandUS/README.md
 * @Description: 
    * 
-->
# FreehandUS
Pose estimation of freedhand ultrasound probe under a semi-structured environment.

## Realworld settings
Design the camera holder and decide the how to install the camera, which will affect the final field of view. Then, write some codes to stream the images and get the intrinsic/extrinsic and DFOV of the cameras. This section includes:

- Dual camera images streaming
  - realworld_settings/multicam_cfg.py
  - realworld_settings/multicam_stream.py
  - realworld_settings/multicam_test.py

- camera intrinsic and extrinsic parameters calibration, the calibration boards can be found in "assets/calibration_board". To generate the calibration board, we suggest: https://calib.io/pages/camera-calibration-pattern-generator .
  - realworld_settings/cam_intrinsic_cali.py
  - realworld_settings/eyeinhand_utils.py & realworld_settings/eyeinhand_cali.py

## Simulation settings
Simulation images and pose pair collection, the simulation scene in coppeliaSim can be found in "assets/coppeliaSim_ttt". To synthesize a dataset in the simulation environment, we should first synchronize the calibration results into the simulation settings and then write a script to communicate with coppeliasim.
- Preliminaries
  
  - copy and paste some files
       - "remoteApi.so" if you are using ubuntu 20.04
       - sim.py
       - simConst.py

  - open coppeliaSim
  entering the following commands, and import the scene (*.ttt) in the "vrep_sim/scenes/xxxx" into the CoppeliaSim 
      ```
      $ cd Downloads/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04/
      $ ./coppeliaSim.sh 
      ```

  - coppeliaSim setting
  define an object as access to the python script. Associate a "Child script" to this object, choose "Non-Threading", use "Lua" type, paste "simRemoteApi.start(19999)" into the "function sysCall_init()".

  - start communication
  Start running coppeliaSim first, and then run the python script.
  



