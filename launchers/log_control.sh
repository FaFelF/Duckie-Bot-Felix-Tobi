#!/bin/bash
source /opt/ros/noetic/setup.bash

LOGFILE=~/Pictures/DuckieRace/control_log_$(date +%Y%m%d_%H%M%S).txt

echo "Logging error + v + omega to $LOGFILE"
echo "Press Ctrl+C to stop."

rostopic echo /gundel/detect/lane          | grep "data:" | sed 's/data:/error:/' >> $LOGFILE &
rostopic echo /gundel/debug/control_v      | grep "data:" | sed 's/data:/v:/'     >> $LOGFILE &
rostopic echo /gundel/debug/control_omega  | grep "data:" | sed 's/data:/omega:/' >> $LOGFILE &

wait
