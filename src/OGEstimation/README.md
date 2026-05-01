# How to run localization in slam mode (this is the best you've got...)

Preface... this is all going to work much much better if you run this on a local network...
    1. launch a hotspot on a laptop natively running ubuntu 22.04 (cell phone's don't seem to work)
        a. wifi settings -> three dots -> turn on wifi hotspot
    2. configure the pi to connect to the hotspot
    3. If you want internet access in this mode, connect a iphone over usb to the computer hosting the hotspot. Ensure the hotspot is enabled

    In theory, this should allow you to run nodes on other devices...



1. Launch the wasd node (ros2 run clanker_hardware wasd.py)
    a. set the tune mode param (ros2 param set wasd_node tune_mode false)
    b. set the pwm mode param (ros2 param set wasd_node pwm_mode false)
2. Launch the rover node (ros2 launch robo_rover rover_launch.py)
    a. wait for the gyro to finish calibrating - check that it is done correctly by (ros2 topic echo /imu/gyro). The anuglar z should be zero if the rover is on the ground
3. Run the slam (ros2 launch estimatation_mapping estimation_launch.py)
4. Now you need to map, drive the robot slowly around the hallway, You need to stop at each "feature" ... hallway ... to let the map update.
    a. make sure you do slightly over 1 lap, you should see loop closure (typically takes approx 1.25 laps)
    b. if you want it to be good run two laps...
5. now you may echo the pose ros2 run tf2_ros tf2 echo map base_link
    a. The pose of the robot is a tf2 frame not a topic
6. kill the wasd node...
    a. now you may run the controller (ros2 run clanker_controls pursuit_node.py)
    b. you need to figure it out from here...

You need to map because loading the map into slam toolbox doesn't work (this also prohibits pure localization mode)

4/18 notes -> make sure you pull controls and robo rover
    -> if you have pixhawk connection issues, look at the readme...
    -> Estimation -> Mapping
    -> Controls -> tf2-pose
    -> RoboRover -> resolve-conflicts (If you are feeling ambitious you can run manual test - I merged some code in that Kelvin added, though the merge is completely untested)
        -> could be best to leave this alone... i dunno

    Control culprit #1 -> look how I'm setting the current velocity estimate in control_cb. tf doesn't handle state derivatives so I need to source that from other places.
        -> I was really hesitant to take a numerical derivative due to the jumpy nature of the slam estimate (I you do, I'd expect a lot of logic would be required to protect it)
        -> I was also hesistant to fix it to tie it to the commanded velocity (that would be the next thing I try)
        -> eneded up tying to ol_rates -> effectively should fix it to zero in not pwm mode (this would work in pwm mode but don't to that cuase slam doesn't like it :( )

    DISCLAIMER -> I patched the pure pursuit node late last night and never tested on the rover -> from the simulations I did remotely, it doesn't crash and collects transforms, however, I expect on the real system you may encounter more challenges. Sry ig

GOOD LUCK


# Estimation
Estimation subsystem for ClankerCollective.

```
ros2 launch robo_rover rover_launch.py
ros2 launch rplidar_ros rplidar_a1_launch.py
ros2 run tf2 static_transform_publisher -0.0251 0.0 0.1683 0 0 0
base_link laser
ros2 launch slam_toolbox online_async_launch.py
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=ClankerCollective/src/Autonomy/rrt_planning/Course11.yaml
rviz2
ros2 run tf2_ros tf2_echo map odom\
\
```