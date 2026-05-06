#!/bin/bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash

rosrun explore_duckietown_ii configuration_node.py &
rosrun explore_duckietown_ii detect_lane_node.py &
rosrun explore_duckietown_ii detect_intersection_node.py &
rosrun explore_duckietown_ii switch_control_node.py &
rosrun explore_duckietown_ii debug_view_node.py &
sleep 5

rosrun explore_duckietown_ii control_lane_node.py &
wait
