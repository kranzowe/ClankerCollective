from setuptools import find_packages, setup

package_name = 'camera_test_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/camera_test_pkg']),
    ('share/camera_test_pkg', ['package.xml']),
    ('share/camera_test_pkg/launch',
        ['launch/camera_tracking.launch.py']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clanker',
    maintainer_email='clanker@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'my_color_sub_opencv = camera_test_pkg.my_color_sub_opencv:main'
        ],
    },
)
