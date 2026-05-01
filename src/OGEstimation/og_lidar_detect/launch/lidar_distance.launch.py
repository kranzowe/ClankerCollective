#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rplidar_ros'),
                'launch',
                'view_rplidar_a1_launch.py'
            )
        )
    )

    wall_detector_node = Node(
        package='og_lidar_detect',
        executable='wall_detector',
        name='wall_detector_node',
        output='screen'
    )

    return LaunchDescription([
        rplidar_launch,
        wall_detector_node
    ])