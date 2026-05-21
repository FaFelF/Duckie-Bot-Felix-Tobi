cmake_minimum_required(VERSION 2.8.3)
project(explore_duckietown_ii)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  duckietown_msgs
)

add_message_files(
  FILES
  DuckieDetection.msg
  DuckieDetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  duckietown_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy message_runtime std_msgs duckietown_msgs
)
