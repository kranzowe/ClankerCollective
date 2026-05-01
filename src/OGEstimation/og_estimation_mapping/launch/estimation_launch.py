#!/usr/bin/env python3
"""
ROS2 Launch file for Estimation subsystem
Launches SLAM Toolbox and EKF for mapping and state estimation
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory('og_estimation_mapping')
    slam_config = os.path.join(package_share_dir, 'config', 'mapper_params.yaml')
    rplidar_launch = os.path.join(
        get_package_share_directory('rplidar_ros'),
        'launch',
        'rplidar_a1_launch.py',
    )
    urdf_file = os.path.join(package_share_dir, 'urdf', 'simple.urdf')

    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content
        }],
    )

    rplidar_launch_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rplidar_launch)
    )

    # SLAM Toolbox node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_config],
    )

    # Delay SLAM startup to allow rover node to initialize first
    delayed_slam = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='Starting slam_toolbox after 5 second delay'),
            slam_toolbox_node,
        ]
    )

    log_info = LogInfo(
        msg=[
            'Launching Estimation subsystem:\n',
            '  URDF: ', urdf_file, '\n',
            '  RPLidar launch: ', rplidar_launch, '\n',
            '  SLAM config: ', slam_config, '\n',
        ]
    )

    return LaunchDescription([
        log_info,
        robot_state_publisher_node,
        rplidar_launch_action,
        delayed_slam,
    ])
