cmake_minimum_required(VERSION 3.0.2)

# EDIT: name of package
project(evetac_software)

# EDIT: list all <build_depend> and <depend> package dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  sensor_msgs
)

catkin_python_setup()

add_message_files(
  DIRECTORY msg 
  FILES EBFloats.msg 
  EBInt16.msg 
  EBUInt16.msg 
  EBInt8.msg 
  EBStampedFloats.msg 
  EBStampedInt16.msg 
  EBStampedInt8.msg 
  RobotisGripper.msg 
  EBTimestamp.msg 
  EvetacMsg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
  sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# EDIT: list all executable python and shell scripts for installation
install(PROGRAMS
   scripts/*
   nodes/*
   ebcam-readout/*
   ebcam-visualize/*
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)