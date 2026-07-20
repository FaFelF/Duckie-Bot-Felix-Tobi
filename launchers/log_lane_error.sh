#!/bin/bash
source /opt/ros/noetic/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

mkdir -p /root/DuckieRace/logs
LOGFILE=/root/DuckieRace/logs/lane_log_$(date +%Y%m%d_%H%M%S).txt

echo "Logging lane error to $LOGFILE"
echo "Press Ctrl+C to stop."

rostopic echo /${VEHICLE_NAME}/detect/lane | grep "data:" >> "$LOGFILE"
