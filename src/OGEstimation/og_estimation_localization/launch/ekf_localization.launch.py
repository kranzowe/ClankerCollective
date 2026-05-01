from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    localization_pkg = get_package_share_directory('estimation_localization')
    mapping_pkg = get_package_share_directory('estimation_mapping')

    default_map = os.path.join(localization_pkg, 'maps', 'Course11.yaml')
    ekf_params = os.path.join(localization_pkg, 'config', 'ekf_localization.yaml')
    lifecycle_params = os.path.join(localization_pkg, 'config', 'lifecycle_slam.yaml')
    default_rviz = os.path.join(localization_pkg, 'rviz', 'ekf_localization.rviz')
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
        condition=IfCondition(LaunchConfiguration('start_lidar')),
    )

    odom_tf_node = Node(
        package='estimation_localization',
        executable='odom_tf_publisher',
        name='odom_tf_publisher',
        output='screen',
    )

    # map_server is here only so RViz can render the map; the EKF loads its own
    # cloud directly from the YAML for ICP.
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_yaml,
            'use_sim_time': False,
        }]
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[lifecycle_params],
    )

    # Delay EKF startup so /odom + odom->base_link TF are flowing first.
    ekf_node = TimerAction(
        period=2.0,
        actions=[Node(
            package='estimation_localization',
            executable='ekf_localization_node',
            name='ekf_localization',
            output='screen',
            parameters=[
                ekf_params,
                {'map_yaml': map_yaml},
            ],
        )],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Full path to map yaml file (used by both map_server and the EKF cloud loader).',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz with the pre-configured EKF visualization.',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz,
            description='Full path to the RViz config file.',
        ),
        DeclareLaunchArgument(
            'start_lidar',
            default_value='true',
            description='Bring up the rplidar driver. Set false if it is already running.',
        ),
        robot_state_publisher_node,
        rplidar_launch_action,
        map_server_node,
        lifecycle_manager_node,
        odom_tf_node,
        ekf_node,
        rviz_node,
    ])
