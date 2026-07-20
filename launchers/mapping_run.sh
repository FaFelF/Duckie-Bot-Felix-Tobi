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

CONFIG_DIR="$(rospack find explore_duckietown_ii)/config"
MAP="$CONFIG_DIR/known_map.json"

# Auswahl aus der Karte holen statt raten zu lassen (B hat keine Ausfahrt 4, C keine 3)
map_query() { python3 -c "
import json,sys
d=json.load(open('$MAP'))
if sys.argv[1]=='nodes':
    print(' '.join(sorted(d['nodes'])))
else:
    n=d['nodes'].get(sys.argv[2])
    print(' '.join(sorted(k for k,v in n['exits'].items() if v is not None)) if n else '')
" "$@"; }

contains() { local n="$1"; shift; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }

START_NODE="$1"
START_EXIT="$2"

NODES=$(map_query nodes)
while ! contains "$START_NODE" $NODES; do
    [ -n "$START_NODE" ] && echo "  '$START_NODE' gibt es nicht."
    read -r -p "Startkreuzung [$NODES]: " START_NODE
done

EXITS=$(map_query exits "$START_NODE")
while ! contains "$START_EXIT" $EXITS; do
    [ -n "$START_EXIT" ] && echo "  Kreuzung $START_NODE hat keine Ausfahrt '$START_EXIT'."
    read -r -p "Startausfahrt an $START_NODE [$EXITS]: " START_EXIT
done

echo "Berechne Mapping-Plan: Start an Kreuzung $START_NODE, Ausfahrt $START_EXIT ..."
rosrun explore_duckietown_ii compute_mapping_plan.py \
    --known-map "$MAP" \
    --start-node "$START_NODE" \
    --start-exit "$START_EXIT" \
    --output "$CONFIG_DIR/plan.json" \
    || exit 1

roslaunch explore_duckietown_ii explore.launch run_mode:=mapping \
    map_path:="$MAP" \
    run_label:="Mapping"
