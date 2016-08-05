# SeqSLAM for Tool-Point Positioning

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
