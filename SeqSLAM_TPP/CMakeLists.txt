cmake_minimum_required(VERSION 2.8.3)
project(seqslam_tpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -DBT_USE_DOUBLE_PRECISION -Wall")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  baxter_core_msgs
  roscpp
  image_transport
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  message_filters
  pcl_ros
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  message_generation
) 

find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(PCL REQUIRED)

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################


# add_message_files(
#   DIRECTORY msg
#   FILES
#   EndEffectorCommand.msg
# )


add_message_files(
  FILES
  BoundingBox.msg
)
add_service_files(
  FILES
  UserSelection.srv
  MATLABSrv.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   baxter_core_msgs
# )



###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS baxter_core_msgs cv_bridge roscpp image_transport message_filters moveit_core moveit_ros_planning moveit_ros_planning_interface message_generation
  # INCLUDE_DIRS include
  #  LIBRARIES seqslam_tpp_test
  # DEPENDS system_lib
)

###########

## Build ##
###########
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME}
  SOURCES
  ${FILES_LIST}
)
## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(seqslam_tpp_test
#   src/${PROJECT_NAME}/seqslam_tpp_test.cpp
# )

## Declare a cpp executable
 add_executable(seqslam_tpp src/seqslam_tpp.cpp)
 add_dependencies(seqslam_tpp seqslam_tpp_generate_messages_cpp)
 # add_executable(spawn src/spawn.py)


## Add cmake target dependenciesolour_space.h"
#include <c of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(seqslam_tpp_test_node seqslam_tpp_test_generate_messages_cpp)

## Specify libraries to link a library or executable target against
 target_link_libraries(seqslam_tpp
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES}
 )
 ##target_link_libraries(seqslam_tpp_kinect
 ##  ${catkin_LIBRARIES}
 ##  ${OpenCV_LIBS}
## )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
 install(PROGRAMS
   src/spawn.py
   src/rec_bag.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

# Mark executables and/or libraries for installation
 install(TARGETS seqslam_tpp
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_seqslam_tpp_test.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
