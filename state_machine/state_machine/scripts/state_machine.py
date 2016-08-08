#! /usr/bin/env python
import rospy
import smach
import smach_ros
import sys

import threading

from abortState import AbortState
from moveRobotToNamedPose import MoveRobotToNamedPose
from moveRobotToPose import MoveRobotToPose
from moveRobotToRelativePose import MoveRobotToRelativePose
from publisherState import PublisherState
from toggleBinFillersAndTote import ToggleBinFillersAndTote
from generatePose import GeneratePose
from userInputRequest import UserInputRequest
from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import Pose, Point, Quaternion

# =============================================================================
if __name__ == '__main__':

    rospy.init_node('seqslam_tpp_state_machine')

    which_robot = rospy.get_param('/seqslam_tpp/robot','baxter')

    robot_information = rospy.param('/seqslam_tpp/'+ which_robot)

    initial_pose = paramToPose(robot_information['initial_pose'])
    secondary_pose = paramToPose(robot_information['secondary_pose'])

    initial_noise = (robot_information['initial_noise']['x'],robot_information['initial_noise']['y'],robot_information['initial_noise']['z'])
    secondary_noise = (robot_information['secondary_noise']['x'],robot_information['secondary_noise']['y'],robot_information['secondary_noise']['z'])

    if which_robot == 'baxter':
        # determine which baxter arm is to be used
        movegroup = robot_information['which_arm']

        if movegroup[:4] == 'left':
            limb = 'left_'
        else:
            limb = 'right_'

    elif which_robot == 'ur5':
        movegroup = 'manipulator'

    else:
        rospy.logwarn('Incorrect robot specified')
        sys.exit()

    sm_init = smach.StateMachine(outcomes=['repeat','user_input','abort_next_trial'],
        output_keys=['initial_pose','data'])

    with sm_init:

        sm_init.add('set_the_shelf_collision_scene_init',
            ToggleBinFillersAndTote(action='all_bins'),
               transitions={'succeeded':'generate_initial',
                'failed': 'repeat'})

        sm_init.add('generate_initial',GeneratePose(pose=initial_pose,noise=initial_noise,tag='initial_pose'),
                transitions={'suceeded': 'move_to_initial',
                            'failed':'generate_initial'})

        sm_init.add('move_to_initial', MoveRobotToPose(movegroup=movegroup),
               transitions={'succeeded':'user_input',
                            'failed': 'repeat'})

    sm_userinput = smach.StateMachine(outcomes=['repeat','positioning','abort_next_trial'],input_keys=['initial_pose','data'],
        output_keys=['initial_image','secondary_pose','data'])

    with sm_userinput:

        sm_userinput.add('user_input_request', UserInputRequest(), transitions={'succeeded':'generate_secondary','failed':'repeat'})

        sm_userinput.add('generate_secondary',GeneratePose(pose=secondary_pose,noise=secondary_noise,tag='secondary_pose'),transitions={'succeeded':'move_to_secondary','failed':'generate_secondary'})

        sm_userinput.add('move_to_secondary', MoveRobotToPose(movegroup=movegroup),
        transitions={'succeeded':'positioning',
            'failed': 'repeat'})


    sm_positioning = smach.StateMachine(outcomes=['repeat','grasping',
        'abort_next_object'],input_keys=['next_item_to_pick'],
        output_keys=['googlenet'])

    with sm_positioning:

        sm_positioning.add('capture_image', UserInputRequest(), transitions={'succeeded':'seqslam','failed':'repeat'})

        sm_positioning.add('seqslam', SeqSLAMState(), transitions={'succeeded': '', 'failed':''})

        sm_positioning.add('calculate_move', CalculateMove(), transitions={'succeeded': '', 'failed':''})

        # RelativePose or AbsolutePose?
        sm_userinput.add('move_to_secondary', MoveRobotToRelativePose(movegroup=movegroup,
                                pose_frame_id='/shelf',
                                relative_pose=
                                    Pose(position=Point(-0.10,0,0),
                                        orientation=Quaternion(0,0,0,1)),
                                velocity_scale=1.0),
               transitions={'succeeded':'user_input',
                            'failed': 'repeat'})


    # TOP LEVEL STATE MACHINE
    sm = smach.StateMachine(outcomes=['succeeded', 'aborted'])

    with sm:
        sm.add('INITIAL', sm_init,  transitions={'repeat':'INITIAL','user_input':'USERINPUT','abort_next_trial':'ABORT'},remapping={'initial_pose':'initial_pose'})
        sm.add('USERINPUT', sm_userinput, transitions={'repeat':'USERINPUT','positioning':'POSITIONING','abort_next_trial':'ABORT'},remapping={'initial_image':'initial_image','secondary_pose':'secondary_pose', 'secondary_image':'secondary_image'})
        sm.add('POSITIONING', sm_positioning, transitions={'repeat':'POSITIONING','grasping':'GRASPING','abort_next_trial':'ABORT'},remapping={'data':'data'})
        sm.add('SAVEDATA',sm_savedata,transitions={'repeat':'SAVEDATA','initial':'INITIAL','abort_next_trial':'ABORT'})


        sm.add('ABORT', AbortState(),
              transitions={'succeeded':'INITIAL'})


    # Create and start the introspection server
    #  (smach_viewer is broken in indigo + 14.04, so need for that now)
    sis = smach_ros.IntrospectionServer('server_name', sm, '/NASA_SM')
    sis.start()

    # run the state machine
    #   We start it in a separate thread so we can cleanly shut it down via CTRL-C
    #   by requesting preemption.
    #   The state machine normally runs until a terminal state is reached.
    smach_thread = threading.Thread(target=sm.execute)
    smach_thread.start()
    # sm.execute()

    # Wait for ctrl-c to stop the application
    rospy.spin()

    # request the state machine to preempt and wait for the SM thread to finish
    sm.request_preempt()
    smach_thread.join()

    # stop the introspection server
    sis.stop()

def paramToPose(inputInfo=None):

    outputPose = Pose()

    outputPose.position = Point(inputInfo['position']['x'],inputInfo['position']['y'],inputInfo['position']['z'])

    quaternion = quaternion_from_euler(inputInfo['orientation']['roll'],inputInfo['orientation']['pitch'],inputInfo['orientation']['yaw'])

    outputPose.orientation = Quaternion(*quaternion)

    return outputPose
