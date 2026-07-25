#!/bin/bash
# Kreuzungs-Lauf (Challenge 2): folgt der Spur, haelt an der roten Linie, richtet sich aus und
# biegt dann zufaellig in eine vom Kreuzungstyp-Tag erlaubte Richtung ab. Kein Plan, kein
# Dashboard -- die Richtung kommt live aus dem AprilTag (direction_mode:=random_tag).
#
# Aufruf: launchers/intersection_run.sh

source /opt/ros/noetic/setup.bash
source /root/DuckieRace/devel/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

roslaunch explore_duckietown_ii explore.launch \
    direction_mode:=random_tag \
    run_label:="Kreuzungen"
