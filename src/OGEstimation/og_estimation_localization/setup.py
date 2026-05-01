from setuptools import setup
from glob import glob
import os

package_name = 'og_estimation_localization'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clanker',
    maintainer_email='clanker@example.com',
    description='Localization package for AMCL and slam_toolbox localization',
    license='Apache-2.0',
    tests_require=['pytest'],
    extras_require={
        'test': ['pytest', 'numpy', 'scipy', 'pillow', 'pyyaml'],
    },
    entry_points={
        'console_scripts': [
            'odom_tf_publisher = og_estimation_localization.odom_tf_publisher:main',
            'ekf_localization_node = og_estimation_localization.ekf_localization_node:main',
        ],
    },
)