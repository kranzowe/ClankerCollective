from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'og_lidar_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clanker',
    maintainer_email='kelvin.aladum@colorado.edu',
    description='Lidar front distance detector',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_detector = og_lidar_detect.detector:main'
        ],
    },
)