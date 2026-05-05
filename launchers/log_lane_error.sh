#!/bin/bash
source /opt/ros/noetic/setup.bash

LOGFILE=~/Pictures/DuckieRace/lane_log_$(date +%Y%m%d_%H%M%S).txt

echo "Logging lane error to $LOGFILE"
echo "Press Ctrl+C to stop."

rostopic echo /gundel/detect/lane | grep "data:" >> $LOGFILE
