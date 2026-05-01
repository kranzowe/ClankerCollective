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


# with C++ image version:

1.ros2 packages:
```bash
colcon build --packages-sekect camera_cpp_pkg
source install/setup.bash
ros2 run camera_cpp_pkg image_listener
```

```bash
ros2 launch camera_test_pkg camera_tracking.launch.py
ros2 launch robo_rover rover_launch.py
```

2. Start line follower:

At \Clankercollective\src\camera_test_pkg\camera_test_pkg\camera_test_pkg run:

`python3 pid_line_follow_closedCV.py`

To finetune the speed as a ros2 parameter:

`ros2 param set /line_follower_node default_speed -1395.0`

3. Start stopsign:


`python3 my_color_sub_opencv_stopsign.py`

`python3 pid_line_follow_stopsign.py`

4. slidingcontrol:
```bash
python3 my_color_sub_closedCV.py
python3 slidingcontrol_line_follow.py
```

some testing: simulation of picture to ros2 node

`python3 publish_image.py`
