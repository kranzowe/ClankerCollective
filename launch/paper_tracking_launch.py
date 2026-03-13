from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Color tracking for Hardware Check-In 3"""
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'pointcloud.enable': 'false',
            'rgb_camera.color_profile': '1920x1080x15'
        }.items()
    )
    color_tracking_node = Node(
        package='camera_test_pkg',
        executable='my_color_sub',
        name='color_tracker'
    )
    targeting_node = Node(
        package='targeting',
        executable='paper_targeting',
        name='control_target'
    )
    return LaunchDescription([
        realsense_launch,
        color_tracking_node,
        targeting_node
    ])
