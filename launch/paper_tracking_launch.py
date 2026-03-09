from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_test_pkg',
            namespace='camera_test_pkg',
            executable='my_color_sub',
            name='blob_rectangle'
        ),
        Node(
            package='targeting',
            namespace='targeting',
            executable='paper_targeting',
            name='control_target'
        ),
    ])