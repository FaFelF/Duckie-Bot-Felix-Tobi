#!/bin/bash
# Mapping-Lauf: berechnet zuerst die Route, die jede Kante der bekannten
# Stadt-Topologie abdeckt, und startet dann den Lauf inkl. mapping_recorder_node
# (ordnet die dabei gesehenen Tor-Tags den Kanten zu -> config/mapped_map.json).
#
# Aufruf: launchers/mapping_run.sh [start_node] [start_exit]
#   start_node: Knoten-ID in known_map.json, an der die Fahrt beginnt (Default: A)
#   start_exit: Ausfahrt 1-4, in die zuerst gefahren wird (Default: 1)

source /opt/ros/noetic/setup.bash
source /root/DuckieRace/devel/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

START_NODE="${1:-A}"
START_EXIT="${2:-1}"
CONFIG_DIR="$(rospack find explore_duckietown_ii)/config"

echo "Berechne Mapping-Plan: Start an Kreuzung $START_NODE, Ausfahrt $START_EXIT ..."
rosrun explore_duckietown_ii compute_mapping_plan.py \
    --known-map "$CONFIG_DIR/known_map.json" \
    --start-node "$START_NODE" \
    --start-exit "$START_EXIT" \
    --output "$CONFIG_DIR/plan.json" \
    || exit 1

roslaunch explore_duckietown_ii explore.launch run_mode:=mapping \
    map_path:="$CONFIG_DIR/known_map.json" \
    run_label:="Mapping"
