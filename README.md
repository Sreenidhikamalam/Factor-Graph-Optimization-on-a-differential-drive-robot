# Factor-Graph-Optimization-on-a-differential-drive-robot

This project focuses on the design, development, and localization of a custom differential drive robot using Factor Graph Optimization (FGO) for sensor fusion and state estimation. The robot was integrated with LiDAR and IMU sensors to estimate motion accurately under different operating conditions.
The objective of this work was to analyze the effect of wheel slip on localization accuracy and evaluate the robustness of FGO under both slip and non-slip environments.

**Project Description**

A differential drive mobile robot was designed and developed for autonomous navigation and localization experiments. The robot was equipped with:
1. LiDAR sensor for environmental perception and scan-based motion estimation
2. IMU sensor for orientation and acceleration measurements
3. Differential drive wheel configuration for motion control

Sensor data from the LiDAR and IMU were collected and fused using Factor Graph Optimization (FGO) techniques to improve localization accuracy.

**Experimental Setup**

Two separate experiments were conducted to evaluate robot performance:

1. Non-Slip Condition

The robot was operated on a normal surface with minimal wheel slippage. Sensor data from LiDAR and IMU were collected while the robot followed a predefined trajectory.

2. Slip Condition

The robot was tested on a low-friction surface where wheel slip was intentionally introduced. This experiment was conducted to study the degradation in odometry accuracy and analyze how FGO improves localization under challenging conditions.

For both experiments:

Ground truth trajectories were manually recorded and compared with estimated trajectories.
Sensor fusion results were analyzed using experimental datasets.
Localization performance was evaluated under varying motion conditions.

**Methodology**

The localization pipeline includes:

1. LiDAR Data Acquisition
Collecting scan data for motion estimation and environmental mapping.
2. IMU Data Processing
Extracting acceleration and orientation information.
3. Sensor Fusion
Combining LiDAR and IMU measurements using Factor Graph Optimization.
4. Trajectory Estimation
Estimating robot pose over time and comparing it against ground truth data.
5. Performance Evaluation
Analyzing localization accuracy in slip and non-slip conditions.

**Repository Contents**
1. fgo.py – Implementation of Factor Graph Optimization and sensor fusion
2. lidar_3.py – LiDAR data acquisition and processing
3. arduino_comb.ino – Differential drive robot control code
4. sensor_fusion_expt1.csv – Experimental dataset for non-slip condition
5. sensor_fusion_expt2.csv – Experimental dataset for slip condition

**Results**

The experiments demonstrated that:
1. Wheel slip significantly affects odometry-based localization accuracy.
2. Factor Graph Optimization improves trajectory estimation by combining multiple sensor measurements.
3. LiDAR and IMU fusion provides more robust localization compared to standalone odometry.
4. The optimized trajectory remained closer to the ground truth even under slip conditions.
