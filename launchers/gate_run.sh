#!/bin/bash
# Torlauf: berechnet zuerst die Route durch die vorgegebene Tor-Reihenfolge auf
# Basis des bereits gemappten Graphen (config/mapped_map.json, Ergebnis des
# Mapping-Laufs), startet dann den Lauf OHNE mapping_recorder_node.
#
# Ohne Argumente fragt das Skript Schritt fuer Schritt nach und zeigt dabei, was
# in der Karte ueberhaupt zur Auswahl steht -- eine Ausfahrt, die es an der
# Kreuzung nicht gibt (B hat keine 4, C keine 3), laesst sich so nicht mehr
# eingeben.
#
# Aufruf: launchers/gate_run.sh [start_node] [start_exit] [tor_reihenfolge]
#   start_node:      Knoten-ID, an der die Fahrt beginnt
#   start_exit:      Ausfahrt 1-4, in die zuerst gefahren wird
#   tor_reihenfolge: Tor-Tags durch Komma getrennt, z.B. 12,7,5

source /opt/ros/noetic/setup.bash
source /root/DuckieRace/devel/setup.bash

[ -z "$VEHICLE_NAME" ] && source "$(cd "$(dirname "$0")/.." && pwd)/duckie-env.sh"

CONFIG_DIR="$(rospack find explore_duckietown_ii)/config"
MAP="$CONFIG_DIR/mapped_map.json"

if [ ! -f "$MAP" ]; then
    echo "FEHLER: $MAP fehlt -- erst launchers/mapping_run.sh fahren."
    exit 1
fi

# kleine Abfragen an die Karte, damit die Auswahl aus der Karte kommt statt geraten zu werden
map_query() { python3 -c "
import json,sys
d=json.load(open('$MAP'))
what=sys.argv[1]
if what=='nodes':
    print(' '.join(sorted(d['nodes'])))
elif what=='exits':
    n=d['nodes'].get(sys.argv[2])
    print(' '.join(sorted(k for k,v in n['exits'].items() if v is not None)) if n else '')
elif what=='tags':
    print(' '.join(str(e['gate_tag']) for e in sorted(d['edges'].values(), key=lambda e:e['id']) if e['gate_tag'] is not None))
" "$@"; }

contains() { local n="$1"; shift; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }

START_NODE="$1"
START_EXIT="$2"
TAG_SEQUENCE="$3"

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

TAGS=$(map_query tags)
while [ -z "$TAG_SEQUENCE" ]; do
    echo "  Gemappte Tore: $TAGS"
    read -r -p "Tor-Reihenfolge (Komma, z.B. ${TAGS// /,}): " TAG_SEQUENCE
done

echo "Berechne Torlauf-Plan: Start an Kreuzung $START_NODE, Ausfahrt $START_EXIT, Tore: $TAG_SEQUENCE ..."
rosrun explore_duckietown_ii compute_gate_plan.py \
    --map "$MAP" \
    --tags "$TAG_SEQUENCE" \
    --start-node "$START_NODE" \
    --start-exit "$START_EXIT" \
    --output "$CONFIG_DIR/plan.json" \
    || exit 1

roslaunch explore_duckietown_ii explore.launch run_mode:=gate \
    map_path:="$MAP" \
    run_label:="Torlauf"
