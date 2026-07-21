#!/bin/bash
# Enten-Ausweichen: startet NUR die dafuer noetigen Nodes (detect_duckies, switch_control,
# control_lane) plus die Debug-/Tuning-Fenster. Kein Kreuzungs-/Tag-/Lane-Node.
#
# Aufruf: launchers/duckie_run.sh            (mit Debug-Fenstern)
#         launchers/duckie_run.sh debug:=false   (nur Duckie-Fenster, sonst headless)

source /opt/ros/noetic/setup.bash
source /root/DuckieRace/devel/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

roslaunch explore_duckietown_ii duckie.launch "$@"
