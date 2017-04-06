# SeqReg for Tool-Point Positioning

[James Sergeant](mailto:james.sergeant@qut.edu.au)

Utilised in

J. Sergeant, G. Doran, D. R. Thompson, C. Lehnert, A. Allwood, B. Upcroft, M. Milford, "Towards Multimodal and Condition-Invariant Vision-based Registration for Robot Positioning on Changing Surfaces," Proceedings of the Australasian Conference on Robotics and Automation 2016 [under review], 2016.

J. Sergeant, G. Doran, D. R. Thompson, C. Lehnert, A. Allwood, B. Upcroft, M. Milford, "Appearance-Invariant Surface Registration for Robot Positioning," International Conference on Robotics and Automation 2017 [under review], 2017.

## Multimodal Dataset
The associated Multimodal Rock Image Dataset can be found [here](https://cloudstor.aarnet.edu.au/plus/index.php/s/nX1rhsKMehp1h6N). After uncompressing to desired location, ensure to specify the path to the dataset when initialising the SeqReg_TPP object in MATLAB. To run the test for all test casees in the dataset:

```
seqreg = SeqReq_TPP();
seqreq.test_dataset();
```

## Other Images
Various input parameter pairs can be provided to both ImagePair() and ImageRegistration(), see next section.
```
image_pair = ImagePair();
image_pair.set_images(image1,image2,relative_scale);
registrator = ImageRegistration()
% methods include 'seqreg','cnn','surf'
results = registrator('method',image_pair);
```

## Parameters
Parameters can be set in `seqreq.parameters`.

## Robot Experiments

### Requirements
ROS Indigo
MATLAB (only tested on R2016a)
MoveIt
Smach
Relies on the the *apc_grasping* and *moveit_lib* ROS packages from the [ACRV Amazon Picking Challenge](https://github.com/amazon-picking-challenge/team_acrv.git) repository

### MATLAB ROS Modification Steps
This step may be required if the image registration process takes longer than 10 seconds as a part of a ROS Service call. In MATLAB perform the following steps:
1. `edit robotics.ros.ServiceServer`
1. Add `obj.ResponseBuilder.setTimeout(120000);` after line 173. This is a 2 minute timeout.
1. `rehash toolboxcache`
1. Restart MATLAB.

### Generate Custom ROS Service Message for MATLAB
1. Install [ROS Custom Message Generator](http://au.mathworks.com/help/robotics/ug/install-robotics-system-toolbox-support-packages.html)
1. `rosgenmsg('path/to/seqreg_tpp')`
1. Follow instructions provided by the script.

### Robot Configuration
Various robot configuration parameters can be set in state_machine/parameters/global.yaml

### State Machine
Start the state machine with 'roslaunch state_machine_seqreg state_machine'

1. Manipulator will move to an initial position.
1. Operator required to select a region of interest on the surface in the image.
1. Manipulator moves to secondary position.
1. Image registration.
1. Manipulator moves to estimated required position.
1. Repeats previous 2 steps until completion criteria met or failed image registration.
