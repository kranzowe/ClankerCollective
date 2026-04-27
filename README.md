# ClankerCollective
Advanced robotics

## Repository Structure

This is the integration repo that ties together all subsystems. Each subsystem is a separate Git repository included as a submodule.

```
ClankerCollective/
├── subsystems/
│   ├── Autonomy/    — Autonomy subsystem
│   ├── Estimation/  — Estimation subsystem
│   ├── Controls/    — Controls subsystem
│   └── Interfaces/  — Custom message/service type definitions
└── ...
```

## Getting Started

Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/kranzowe/ClankerCollective.git
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

## Updating Submodules

Pull latest changes from all subsystems:
```bash
git submodule update --remote --merge
```

# Running Sensor Nodes
Launch file for Lidar and wall distance detector:


`ros2 launch lidar_detect lidar_distance.launch.py`

Lidar

`ros2 launch rplidar_ros view_rplidar_a1_launch.py`

`ros2 topic echo /scan --once`

Lidar subscriber, tells the front wall distance:
  
`ros2 run lidar_detect wall_detector`

To check if Lidar topic is workiing:

`ros2 topic echo /ld_distance`   
IMU

`ros2 run robo_rover rover_node`

`ros2 topic echo /imu/accel`

Camera
Setup for the Realsense Camera: Enabling ONLY RGB @6FPS (saivng computing power)

`
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=false enable_infra1:=false enable_infra2:=false enable_gyro:=false enable_accel:=false pointcloud.enable:=false
`

optional: `rgb_camera.color_profile:=640x480x6`

See the topic list, below is what we want: `\camera\camera\color\image_raw Node name: /camera/camera`

This displays the raw image data, which we need a package to run all of this. I made camera_test_pkg located in src of ClankerCollective. After you run what is above, do the following commands in the base of ClankerCollective:

```
source /opt/ros/humble/setup.bash

source ~/ClankerCollective/install/setup.bash

ros2 run camera_test_pkg my_color_sub
```

# Running mapping & localization

`rviz2`

`ros2 launch robo_rover rover_launch.py`

`ros2 launch estimation_mapping estimation_launch.py`

`ros2 launch estimation_localization amcl_localization.launch.py`



# Teleoperation

`ros2 run robo_rover rover_node`

`ros2 run clanker_hardware wasd.py`


# C++ image version:

publish_image.py: simulation of picture to ros2 node
```bash
python3 publish_image.py
```
2.ros2 packages:
```bash
colcon build --packages-sekect camera_cpp_pkg
source install/setup.bash
ros2 run camera_cpp_pkg image_listener
```
for pid_line_follow.py:
Step 1：tune KP first
```bash
PATH_ANGLE_KP = -170.0
```
not turn: -200 / turn too much: -120


Step 2：tune kD
```bash
PATH_ANGLE_KD = -25.0
```
wobbling: -35 / too slow: -15


Step 3：tune RIGHT_BIAS
```bash
RIGHT_BIAS = -180.0
```
not enough: -220 / good: -180 / overshoot: -140

Step 4: when to stop turn right
```bash
if abs(angle_error) < 0.25:
```

too early: 0.15 / keep turning: 0.3

Step 5：Tune speed
```bash
DEFAULT_SPEED = 1415.0
```

Step 6: timeout
```bash
PATH_TIMEOUT_SEC = 2
```
sometimes stop: 3 / response late: 1
