cmake_minimum_required(VERSION 2.8.3)
project(explore_duckietown_ii)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
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
  CATKIN_DEPENDS rospy std_msgs message_runtime duckietown_msgs
)
