from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():
    localization_pkg = get_package_share_directory('og_estimation_localization')
    mapping_pkg = get_package_share_directory('og_estimation_mapping')

    slam_params = os.path.join(localization_pkg, 'config', 'slam_localization.yaml')
    urdf_file = os.path.join(mapping_pkg, 'urdf', 'simple.urdf')
    rplidar_launch = os.path.join(
        get_package_share_directory('rplidar_ros'),
        'launch',
        'rplidar_a1_launch.py',
    )

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    rplidar_launch_action = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(rplidar_launch),
    )

    
    slam_node = Node(
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        rplidar_launch_action,
        slam_node
    ])