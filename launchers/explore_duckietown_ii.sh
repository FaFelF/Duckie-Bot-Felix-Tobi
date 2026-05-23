#!/bin/bash
source /opt/ros/noetic/setup.bash
source /root/DuckieRace/devel/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

rosrun explore_duckietown_ii configuration_node.py &
rosrun explore_duckietown_ii detect_lane_node.py &
rosrun explore_duckietown_ii detect_intersection_node.py &
rosrun explore_duckietown_ii switch_control_node.py &
rosrun explore_duckietown_ii debug_view_node.py &
sleep 5

rosrun explore_duckietown_ii control_lane_node.py &
wait
