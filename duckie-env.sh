#!/bin/bash
# Usage:
#   source duckie-env.sh           # interaktiv (fragt nach Bot-Name)
#   source duckie-env.sh <name>    # direkt mit Bot-Name

CONTAINER_IP=$(hostname -I | awk '{print $1}')

if [ -n "$1" ]; then
    botname="$1"
else
    read -p "Bot-Name: " botname
fi

if [ -z "$botname" ]; then
    echo "Kein Name angegeben, abgebrochen."
    return 1
fi

export VEHICLE_NAME="$botname"
export ROS_MASTER_URI="http://${botname}.local:11311"
export ROS_IP="$CONTAINER_IP"

echo ""
echo "  VEHICLE_NAME   = $VEHICLE_NAME"
echo "  ROS_MASTER_URI = $ROS_MASTER_URI"
echo "  ROS_IP         = $ROS_IP"
