# ClankerCollective
Advanced robits

## Repository Structure

This is the integration repo that ties together all subsystems. Each subsystem is a separate Git repository included as a submodule.

```
ClankerCollective/
├── subsystems/
│   ├── Autonomy/    — Autonomy subsystem
│   ├── Estimation/  — Estimation subsystem
│   ├── Controls/    — Controls subsystem
│   └── Interfaces/  — Custom message/service type definitions
└── ...
```

## Getting Started

Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/kranzowe/ClankerCollective.git
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

## Updating Submodules

Pull latest changes from all subsystems:
```bash
git submodule update --remote --merge
```

# Running Sensor Nodes
Lidar

`ros2 launch rplidar_ros view_rplidar_a1_launch.py`

`ros2 topic echo /scan`

IMU

`ros2 run robo_rover robo_node`

`ros2 topic echo /imu/accel`

Camera

# Teleoperation

`ros2 run robo_rover robo_node`

`ros2 run clanker_hardware wasd.py`

