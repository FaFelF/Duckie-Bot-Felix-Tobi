#!/bin/bash
source /opt/ros/noetic/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

mkdir -p /root/DuckieRace/logs
LOGFILE=/root/DuckieRace/logs/control_log_$(date +%Y%m%d_%H%M%S).txt

echo "Logging error + v + omega to $LOGFILE"
echo "Press Ctrl+C to stop."

rostopic echo /${VEHICLE_NAME}/detect/lane         | grep "data:" | sed 's/data:/error:/' >> "$LOGFILE" &
rostopic echo /${VEHICLE_NAME}/debug/control_v     | grep "data:" | sed 's/data:/v:/'     >> "$LOGFILE" &
rostopic echo /${VEHICLE_NAME}/debug/control_omega | grep "data:" | sed 's/data:/omega:/' >> "$LOGFILE" &

wait
