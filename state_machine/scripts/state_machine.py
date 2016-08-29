#! /usr/bin/env python
import rospy
import smach
import smach_ros
import sys

import threading

from abortState import AbortState
from moveRobotToNamedPose import MoveRobotToNamedPose
from moveRobotState import MoveRobotState
from moveRobotToRelativePose import MoveRobotToRelativePose
from publisherState import PublisherState
from toggleBinFillersAndTote import ToggleBinFillersAndTote
from generatePose import GeneratePose
from userInputRequest import UserInputRequest
from tf.transformations import quaternion_from_euler
from getImage import GetImage
from waitState import WaitState
from seqslamState import SeqSLAMState
from getRobotPose import GetRobotPose
from calcScales import CalcScales
from calcMovement import CalcMovement
from checkLimits import CheckLimits
from initData import InitData
from toggleData import ToggleData

from geometry_msgs.msg import Pose, Point, Quaternion


def paramToPose(inputInfo=None):

    outputPose = Pose()

    outputPose.position = Point(inputInfo['position']['x'],inputInfo['position']['y'],inputInfo['position']['z'])

    quaternion = quaternion_from_euler(inputInfo['orientation']['roll'],inputInfo['orientation']['pitch'],inputInfo['orientation']['yaw'])

    outputPose.orientation = Quaternion(*quaternion)

    return outputPose

# =============================================================================
if __name__ == '__main__':

    rospy.init_node('seqslam_tpp_state_machine')

    which_robot = rospy.get_param('/seqslam_tpp/robot','baxter')

    robot_information = rospy.get_param('/seqslam_tpp/'+ which_robot)

    limits = rospy.get_param('/seqslam_tpp/limits')

    initial_pose = paramToPose(robot_information['initial_pose'])
    secondary_pose = paramToPose(robot_information['secondary_pose'])
    frame_id = robot_information['frame_id']
    sample_distance = robot_information['sample_distance']
    sample_direction = robot_information['sample_direction']
    camera_information = rospy.get_param('/seqslam_tpp/' + rospy.get_param('/seqslam_tpp/camera'))
    method = rospy.get_param('/seqslam_tpp/method')

    initial_noise = (robot_information['initial_noise']['x'],robot_information['initial_noise']['y'],robot_information['initial_noise']['z'])
    secondary_noise = (robot_information['secondary_noise']['x'],robot_information['secondary_noise']['y'],robot_information['secondary_noise']['z'])

    sample_noise = robot_information['sample_noise']

    movegroup = robot_information['move_group']

    if which_robot == 'baxter':

        if movegroup[:4] == 'left':
            limb = 'left_'
        else:
            limb = 'right_'

    else:
        rospy.logwarn('Incorrect robot specified')
        sys.exit()

    sm_init = smach.StateMachine(outcomes=['repeat','user_input','abort_next_trial'],
        output_keys=['data'])

    with sm_init:

        sm_init.add('init_data', InitData(sample_distance=sample_distance, sample_direction=sample_direction, sample_noise=sample_noise, limits=limits, camera_information=camera_information),transitions={'succeeded':'set_the_shelf_collision_scene_init','failed':'repeat'})

        sm_init.add('set_the_shelf_collision_scene_init',
            ToggleBinFillersAndTote(action='all_bins'),
               transitions={'succeeded':'lip_off',
                'failed': 'repeat'})

        sm_init.add('lip_off',
                ToggleBinFillersAndTote(action='lip_off'),
                   transitions={'succeeded':'generate_initial',
                    'failed': 'repeat'})

        sm_init.add('generate_initial',GeneratePose(pose=initial_pose,noise=initial_noise,frame_id=frame_id,tag='initial_pose'),
                transitions={'succeeded': 'move_to_initial',
                            'failed':'generate_initial'})

        sm_init.add('move_to_initial', MoveRobotState(movegroup=movegroup),
               transitions={'succeeded':'user_input',
                            'failed': 'repeat','aborted':'repeat'})

    sm_userinput = smach.StateMachine(outcomes=['repeat','positioning','abort_next_trial'],input_keys=['data'],
        output_keys=['data'])

    with sm_userinput:

        sm_userinput.add('user_input_request', UserInputRequest(action='user_request_initial'), transitions={'succeeded':'get_robot_pose','failed':'repeat','abort':'repeat'})

        sm_userinput.add('get_robot_pose',
            GetRobotPose(movegroup=movegroup,frame_id=frame_id),
            transitions={'succeeded':'generate_secondary','failed':'repeat','aborted':'abort_next_trial'})

        sm_userinput.add('generate_secondary',GeneratePose(pose=secondary_pose,noise=secondary_noise,frame_id=frame_id,tag='secondary_pose'),transitions={'succeeded':'move_to_secondary','failed':'generate_secondary'})

        sm_userinput.add('move_to_secondary', MoveRobotState(movegroup=movegroup + '_cartesian'),
            transitions={'succeeded':'wait_1',
                    'failed': 'repeat','aborted':'repeat'})

        sm_userinput.add('wait_1',WaitState(2.0),transitions={'succeeded':'positioning','preempted':'positioning'})


    sm_positioning = smach.StateMachine(outcomes=['repeat','servoing',
        'abort_next_trial'],input_keys=['data'],
        output_keys=['data'])

    with sm_positioning:

        sm_positioning.add('get_image', GetImage(tag='secondary_image'),
            transitions={'succeeded':'get_robot_pose','failed':'repeat','aborted':'abort_next_trial'})

        sm_positioning.add('get_robot_pose',
                GetRobotPose(movegroup=movegroup,frame_id=frame_id,calculate_scales=True),
                transitions={'succeeded':'calc_scales','failed':'repeat','aborted':'abort_next_trial'})

        sm_positioning.add('calc_scales',
                CalcScales(), transitions={'succeeded':'seqslam','failed':'repeat','aborted':'abort_next_trial'})

        sm_positioning.add('seqslam', SeqSLAMState(method=method), transitions={'succeeded': 'check_limits', 'failed':'seqslam', 'aborted':'abort_next_trial'})

        sm_positioning.add('check_limits', CheckLimits(), transitions={'incomplete': 'calculate_move', 'complete': 'calculate_move_complete', 'aborted':'abort_next_trial'})


        ## LIMITS NOT MET
        sm_positioning.add('calculate_move', CalcMovement(), transitions={'succeeded': 'move', 'failed':'calculate_move','aborted':'abort_next_trial'})

        sm_positioning.add('move', MoveRobotState(movegroup=movegroup),
            transitions={'succeeded':'toggle_servoing_flag',
                    'failed': 'abort_next_trial','aborted':'abort_next_trial'})

        sm_positioning.add('toggle_servoing_flag', ToggleData(key='servoing',value=True),transitions={'succeeded': 'wait_1'})

        sm_positioning.add('wait_1',WaitState(5.0),transitions={'succeeded':'repeat','preempted':'repeat'})

        ## LIMITS MET
        sm_positioning.add('calculate_move_complete', CalcMovement(), transitions={'succeeded': 'move_complete', 'failed':'calculate_move_complete','aborted':'abort_next_trial'})

        sm_positioning.add('move_complete', MoveRobotState(movegroup=movegroup),
            transitions={'succeeded':'wait_1_complete',
                    'failed': 'abort_next_trial','aborted':'abort_next_trial'})

        sm_positioning.add('wait_1_complete',WaitState(5.0),transitions={'succeeded':'get_image_complete','preempted':'get_image_complete'})

        sm_positioning.add('get_image_complete', GetImage(tag='secondary_image'),
            transitions={'succeeded':'get_robot_pose_complete','failed':'repeat','aborted':'abort_next_trial'})

        sm_positioning.add('get_robot_pose_complete',
                GetRobotPose(movegroup=movegroup,frame_id=frame_id,calculate_scales=True),
                transitions={'succeeded':'abort_next_trial','failed':'abort_next_trial','aborted':'abort_next_trial'})

    sm_servoing = smach.StateMachine(outcomes=['repeat','servoing',
            'abort_next_trial'],input_keys=['next_item_to_pick'],
            output_keys=['googlenet'])

    with sm_servoing:

        sm_servoing.add('get_image', GetImage(tag='servoing_image'), transitions={'succeeded':'repeat','failed':'repeat','aborted':'abort_next_trial'})


    # TOP LEVEL STATE MACHINE
    sm = smach.StateMachine(outcomes=['succeeded', 'aborted'])

    with sm:
        sm.add('INITIAL', sm_init,  transitions={'repeat':'INITIAL','user_input':'USERINPUT','abort_next_trial':'ABORT'},remapping={'data':'data'})
        sm.add('USERINPUT', sm_userinput, transitions={'repeat':'USERINPUT','positioning':'POSITIONING','abort_next_trial':'ABORT'},remapping={'data':'data'})
        sm.add('POSITIONING', sm_positioning, transitions={'repeat':'POSITIONING','servoing':'SERVOING','abort_next_trial':'ABORT'},remapping={'data':'data'})
        sm.add('SERVOING', sm_servoing, transitions={'repeat':'SERVOING','servoing':'SERVOING',
                'abort_next_trial':'ABORT'},remapping={'data':'data'})
        # sm.add('SAVEDATA',sm_savedata,transitions={'repeat':'SAVEDATA','initial':'INITIAL','abort_next_trial':'ABORT'})


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
