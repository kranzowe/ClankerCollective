from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    localization_pkg = get_package_share_directory('og_estimation_localization')
    mapping_pkg = get_package_share_directory('og_estimation_mapping')

    default_map = os.path.join(localization_pkg, 'maps', 'lab.yaml')
    slam_params = os.path.join(localization_pkg, 'config', 'slam_localization.yaml')
    lifecycle_params = os.path.join(localization_pkg, 'config', 'lifecycle_slam.yaml')
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

    # Integrates /ol_rates velocities → publishes /odom and odom→base_link TF for slam_toolbox
    odom_tf_node = Node(
        package='og_estimation_localization',
        executable='odom_tf_publisher',
        name='odom_tf_publisher',
        output='screen',
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

    # map_server is a lifecycle node — lifecycle_manager activates it
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[lifecycle_params]
    )

    # Delay SLAM startup so odom TF is being published first
    slam_node = TimerAction(
        period=3.0,
        actions=[Node(
            package='slam_toolbox',
            executable='localization_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params]
        )]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Full path to map yaml file'
        ),
        robot_state_publisher_node,
        rplidar_launch_action,
        map_server_node,
        lifecycle_manager_node,
        odom_tf_node,
        slam_node,
    ])