# SeqSLAM for Tool-Point Positioning

[James Sergeant](james.sergeant@qut.edu.au)

## MATLAB ROS Modification Steps
1. `edit robotics.ros.ServiceServer`
1. Add `obj.ResponseBuilder.setTimeout(120000);` after line 173.
1. `rehash toolboxcache`
1. Restart MATLAB.

## Generate Custom ROS Service Message for MATLAB
1. Install [ROS Custom Message Generator](http://au.mathworks.com/help/robotics/ug/install-robotics-system-toolbox-support-packages.html)
1. `rosgenmsg('path/to/seqslam_tpp')`
1. Follow instructions provided by the script.

## Simulation w/ UR5
Open `. baxter.sh` and run each of the following commands in separate terminals.
```
roslaunch ur_gazebo ur5.launch limited:=true gui:=false
roslaunch harvey_moveit_config harvey_moveit.launch limited:=true sim:=true

```

## Simulation w/ Baxter
Open `. baxter.sh` and run each of the following commands in separate terminals.
```
roslaunch baxter_gazebo baxter_world.launch
rosrun baxter_interface joint_trajectory_action_server.py
rosrun baxter_tools tuck_arms.py -u
roslaunch baxter_moveit_config baxter_moveit.launch
```
