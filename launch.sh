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
    [ -n "$launcher" ] && bash "$SCRIPT_DIR/launchers/$launcher" && break
done
