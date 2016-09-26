cmake_minimum_required(VERSION 2.8.3)
project(user_input)

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


# add_message_files(
#   FILES
#   BoundingBox.msg
# )
add_service_files(
  FILES
  UserSelection.srv
  MATLABSrv.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)



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
  CATKIN_DEPENDS baxter_core_msgs cv_bridge roscpp image_transport message_filters moveit_core moveit_ros_planning moveit_ros_planning_interface message_generation message_runtime sensor_msgs std_msgs
  # INCLUDE_DIRS include
  #  LIBRARIES user_input_test
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

## Declare a cpp executable
 add_executable(user_input src/user_input.cpp)
 add_dependencies(user_input user_input_generate_messages_cpp)

## Specify libraries to link a library or executable target against
 target_link_libraries(user_input
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES}
 )

#############
## Install ##
#############
 install(PROGRAMS
   src/spawn.py
   src/rec_bag.py
   src/test_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

# Mark executables and/or libraries for installation
 install(TARGETS user_input
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
