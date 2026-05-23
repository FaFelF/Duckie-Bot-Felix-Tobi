#!/bin/bash
source /opt/ros/noetic/setup.bash
source /root/DuckieRace/devel/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

rosrun follow_lane detect_lane_node.py &
rosrun follow_lane detect_intersection_node.py &
rosrun follow_lane switch_control_node.py &
sleep 5

rosrun follow_lane control_lane_node.py &
wait
