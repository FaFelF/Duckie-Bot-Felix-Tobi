#!/bin/bash
# Setzt Bot-Umgebung und startet einen Launcher interaktiv.
# Aufruf: bash launch.sh  oder  bash launch.sh <botname>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/duckie-env.sh" "$@"
[ $? -ne 0 ] && exit 1

shopt -s nullglob
launchers=("$SCRIPT_DIR/launchers/"*.sh)

if [ ${#launchers[@]} -eq 0 ]; then
    echo "Keine Launcher gefunden in launchers/"
    exit 1
fi

echo ""
echo "Verfügbare Launcher:"
PS3="Auswahl (Nummer): "
select launcher in "${launchers[@]#$SCRIPT_DIR/launchers/}"; do
    if [ -z "$launcher" ]; then
        echo "Ungueltige Auswahl."
        continue
    fi

    # Manche Launcher brauchen Argumente (gate_run.sh <knoten> <ausgang> <tore>,
    # mapping_run.sh [knoten] [ausgang]). Vorher wurden sie immer ohne aufgerufen und
    # brachen mit "Tor-Reihenfolge fehlt" ab. Die "Aufruf:"-Zeile aus dem Skriptkopf
    # dient als Hilfe.
    usage=$(grep -m1 '^# *Aufruf:' "$SCRIPT_DIR/launchers/$launcher" | sed 's/^# *//')
    [ -n "$usage" ] && echo "  $usage"
    read -r -p "Argumente (leer = keine): " args

    # $args bewusst ohne Anfuehrungszeichen: soll in einzelne Argumente zerlegt werden.
    bash "$SCRIPT_DIR/launchers/$launcher" $args
    break
done
