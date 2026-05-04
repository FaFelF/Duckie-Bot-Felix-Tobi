# Duckie-Befehle Übersicht

Drei Befehle stehen in der VM zur Verfügung, alle aufeinander aufbauend:
`dt` → `dt-launch-sim` → `dt-pick`.

---

## `dt`

**Zweck:** Ein einzelnes Terminal für ROS-Arbeit vorbereiten.

**Was es macht:**
- Sourct `/opt/ros/noetic/setup.bash`
- Sourct `~/DuckieRace_2026/devel/setup.bash` (falls vorhanden)
- Fragt interaktiv: echter Bot oder Simulator?
  - **Echter Bot:** fragt nach Bot-Name → setzt `VEHICLE_NAME`, `ROS_MASTER_URI=http://<name>.local:11311`, `ROS_IP` auf die VM-IP
  - **Simulator:** setzt `VEHICLE_NAME=fakebot`, entfernt `ROS_MASTER_URI` und `ROS_IP`, aktiviert die `dt-sim` Python-Umgebung

**Direktaufruf ohne Menü:**
- `source ~/duckie-env.sh sim` → direkt Sim-Setup
- `source ~/duckie-env.sh bot donald` → direkt Bot-Setup mit Namen `donald`

**Wann benutzen:** In jedem neuen Terminal, das ROS-Befehle ausführen soll (`rosrun`, `roslaunch`, `rostopic`, etc.). Auch in SSH-Sessions von VS Code aus.

**Skript-Datei:** `~/duckie-env.sh`

---

## `dt-launch-sim`

**Zweck:** Den kompletten Simulator-Workflow mit zwei Terminals automatisch starten.

**Was es macht:**
- Öffnet **Terminal 1 ("Sim"):** sourct env (Sim-Modus), startet `roscore` im Hintergrund, wartet 2 Sekunden, startet `python3 sim_wrapper_node.py` aus `~/gym-duckietown/`
- Wartet 4 Sekunden, damit `roscore` voll hochgekommen ist
- Öffnet **Terminal 2 ("Launcher"):** sourct env, wechselt nach `~/DuckieRace_2026`, zeigt Auswahl-Menü aller `.sh`-Dateien aus `launchers/`. Auswahl per Nummer → Skript läuft. Letzter Eintrag „Freies Terminal" überspringt das Menü.

**Wann benutzen:** Wenn du den Simulator-Workflow neu startest. Ein einziger Befehl ersetzt das manuelle Öffnen mehrerer Terminals.

**Wichtig:** Nur in der **VM-GUI** ausführen, nicht über SSH. Die geöffneten Fenster brauchen einen Display.

**Skript-Datei:** `~/sim-launch.sh`

---

## `dt-pick`

**Zweck:** Im aktuellen Terminal das Launcher-Auswahlmenü nochmal aufrufen, ohne neues Fenster zu öffnen.

**Was es macht:**
- Wechselt nach `~/DuckieRace_2026`
- Zeigt Auswahl-Menü aller `.sh`-Dateien aus `launchers/`
- Bei Auswahl: führt das gewählte Skript aus
- Bei „Abbrechen": zurück zur Shell ohne Aktion

**Wann benutzen:** Hauptsächlich nach Code-Änderungen. Workflow:

1. Im Launcher-Terminal `Strg+C` → Skript stoppt, Terminal bleibt offen
2. `catkin_make` → Code neu bauen
3. `source devel/setup.bash` → Build aktivieren
4. `dt-pick` → Launcher neu auswählen und starten

Vorteil gegenüber `dt-launch-sim`: Kein zweiter Sim wird gestartet, der erste läuft weiter im anderen Fenster.

**Definiert in:** `~/.bashrc` (als Funktion)

---

## Typischer Workflow

### Sim-Setup zum ersten Mal

In der VM-GUI ein Terminal öffnen:

```bash
dt-launch-sim
```

Im Launcher-Fenster (T2): Nummer von `follow_lane.sh` wählen → läuft.

### Code geändert, neu testen

In T2:

```bash
# Strg+C drücken um laufendes Skript zu stoppen
catkin_make
source devel/setup.bash
dt-pick
```

→ Auswahl → Neustart.

### Spontan ein viertes Terminal für `rosrun` o.ä.

Neues Terminal in der VM-GUI:

```bash
dt
# Auswahl: 2 (Sim)
rosrun follow_lane configuration_node.py
```

### Echter Bot

Neues Terminal:

```bash
dt
# Auswahl: 1
# Bot-Name: donald
rostopic list
```

---

## Übersicht aller Dateien

| Datei | Zweck |
|---|---|
| `~/duckie-env.sh` | Setup-Skript für ein Terminal (steht hinter `dt`) |
| `~/sim-launch.sh` | Multi-Terminal-Launcher (steht hinter `dt-launch-sim`) |
| `~/.bashrc` | Enthält Aliase und `dt-pick`-Funktion |
| `~/.config/autostart/add-resolution.desktop` | Registriert 3440x1440 als verfügbare Auflösung beim Login |
| `~/.local/bin/add-resolution.sh` | Skript dahinter (registriert den xrandr-Modus) |
