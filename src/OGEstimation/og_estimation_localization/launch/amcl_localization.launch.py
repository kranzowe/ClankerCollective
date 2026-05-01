from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():
    localization_pkg = get_package_share_directory('og_estimation_localization')
    mapping_pkg = get_package_share_directory('og_estimation_mapping')

    default_map = os.path.join(localization_pkg, 'maps', 'lab.yaml')
    amcl_params = os.path.join(localization_pkg, 'config', 'amcl.yaml')
    lifecycle_params = os.path.join(localization_pkg, 'config', 'lifecycle_localization.yaml')
    urdf_file = os.path.join(mapping_pkg, 'urdf', 'simple.urdf')
    rplidar_launch = os.path.join(
        get_package_share_directory('rplidar_ros'),
        'launch',
        'rplidar_a1_launch.py',
    )

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    map_yaml = LaunchConfiguration('map')

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
    

    odom_tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='odom_to_base_link',
    arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
    
    )
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_yaml,
            'use_sim_time': False
        }]
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_params]
    )

    lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[lifecycle_params]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Full path to map yaml file'
        ),
        robot_state_publisher_node,
        map_server_node,
        amcl_node,
        lifecycle_node,
        rplidar_launch_action,
        odom_tf_node
    ])