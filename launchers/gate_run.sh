#!/bin/bash
# Torlauf: berechnet zuerst die Route durch die vorgegebene Tor-Reihenfolge auf
# Basis des bereits gemappten Graphen (config/mapped_map.json, Ergebnis des
# Mapping-Laufs), startet dann den Lauf OHNE mapping_recorder_node.
#
# Aufruf: launchers/gate_run.sh <start_node> <start_exit> <tor_reihenfolge>
#   start_node:      Knoten-ID, an der die Fahrt beginnt
#   start_exit:      Ausfahrt 1-4, in die zuerst gefahren wird
#   tor_reihenfolge: Tor-Tags durch Komma getrennt, z.B. 12,7,5

source /opt/ros/noetic/setup.bash
source /root/DuckieRace/devel/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

START_NODE="${1:-A}"
START_EXIT="${2:-1}"
TAG_SEQUENCE="${3:?Tor-Reihenfolge fehlt, z.B. launchers/gate_run.sh A 1 12,7,5}"
CONFIG_DIR="$(rospack find explore_duckietown_ii)/config"

echo "Berechne Torlauf-Plan: Start an Kreuzung $START_NODE, Ausfahrt $START_EXIT, Tore: $TAG_SEQUENCE ..."
rosrun explore_duckietown_ii compute_gate_plan.py \
    --map "$CONFIG_DIR/mapped_map.json" \
    --tags "$TAG_SEQUENCE" \
    --start-node "$START_NODE" \
    --start-exit "$START_EXIT" \
    --output "$CONFIG_DIR/plan.json" \
    || exit 1

roslaunch explore_duckietown_ii explore.launch run_mode:=gate \
    map_path:="$CONFIG_DIR/mapped_map.json" \
    run_label:="Torlauf"
