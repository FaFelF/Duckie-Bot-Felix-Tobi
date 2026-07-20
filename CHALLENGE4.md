# Challenge 4 — Mapping & Path Finding: Handoff-Dokument

Dieses Dokument fasst die komplette Challenge-4-Architektur zusammen, die auf dem
bestehenden `explore_duckietown_ii`-ROS-Package aufgebaut wurde. Ziel: jemand
(oder eine andere Claude-Instanz), der/die den Code noch nicht kennt, soll hier
alles finden, um ihn zu verstehen, zu erklären und den Bot damit zu fahren.

Alles unter "Neue Dateien" wurde in einer Entwicklungs-Session auf einem
Windows-Rechner **ohne ROS** geschrieben — die komplette Logik wurde als reines
Python getestet (siehe "Teststand" unten), aber **noch nie in echtem ROS
gelaufen**. Das ist der wichtigste Punkt für den nächsten Schritt.

---

## 1. Aufgabenstellung (Kurzfassung)

Der Duckiebot bekommt eine Karte der Stadt als Graph (Kreuzungen = Knoten,
Straßen = Kanten). Ablauf in zwei getrennten Läufen:

1. **Mapping-Lauf:** Bot fährt durchs Straßennetz und ordnet dabei erkannte
   "Tore" (farbige AprilTags, IDs 5–13) den Graph-Kanten zu. AprilTags 1–4
   bleiben wie bisher für Kreuzungstypen reserviert.
2. **Torlauf (zeitkritisch):** Mit dem fertig gemappten Graphen wird eine
   vorgegebene Tor-Reihenfolge abgefahren — der Pfad muss algorithmisch
   berechnet werden, **darf nicht hardgecodet sein**.

Weitere Vorgaben aus der Aufgabenstellung:
- Der Graph ist **ungerichtet** — Spur und Gegenspur tragen denselben Tag.
- Nummerierung der Ausfahrten an jeder Kreuzung ist **fix**: 1 und 3 liegen
  sich gegenüber, 2 ist rechts von 1, 4 ist links von 1 (kompassgebunden,
  gleiche Bedeutung an jeder Kreuzung der Karte).
- Anforderung: Karte, gewählter Pfad und aktuelle Position müssen sichtbar
  gemacht werden (Dashboard).
- Nur der Torlauf ist zeitkritisch — der Pfad muss nicht global optimal sein,
  nur "nicht unnötig lang".
- Startpunkt ist frei wählbar, kann sich zwischen Mapping-Lauf und Torlauf
  unterscheiden.

---

## 2. Zentrale Design-Entscheidungen (bereits mit dem Nutzer abgestimmt)

Diese Punkte wurden im Verlauf der Entwicklung explizit diskutiert und
festgelegt — **bitte nicht neu aufrollen**, außer es gibt einen konkreten Grund:

- **Die Stadt-Topologie ist vor dem Mapping-Lauf bereits bekannt** (aus dem
  Foliensatz / am Prüfungstag gezeigt) — nur welcher Tag auf welcher Kante
  sitzt, ist unbekannt. Es gibt daher **kein SLAM-/Loop-Closure-Problem**: der
  Bot muss keine unbekannte Topologie entdecken, nur eine vorgeplante Route
  abfahren und dabei Tags einsammeln.
- **Hardware-Positionstracking ist nicht möglich.** Es gibt keine Odometrie/
  Lokalisierung. Die "Position" ist rein symbolisch: ein Zähler, der bei jeder
  erkannten Kreuzung hochgezählt wird (Index in einer vorab berechneten
  Hop-Liste, siehe `PlanState` unten).
- **Kein automatischer Konsistenz-Check.** Es wird NICHT geprüft, ob der
  erkannte Kreuzungstyp (Tag 1–4) zum erwarteten Typ laut Graph passt. Bewusste
  Entscheidung des Nutzers: Fehler beim Verpassen einer Kreuzung werden übers
  Dashboard von einem Menschen beobachtet, nicht automatisch abgefangen.
- **Kürzeste Anzahl Kreuzungen statt global kürzestem Weg.** Für Routenplanung
  wird BFS (Kreuzungsanzahl) statt Dijkstra/A* verwendet, bei mehreren gleich
  kurzen Optionen wird zufällig gewählt. Muss nicht optimal sein, nur
  "nicht unnötig lang".
- **Keine aktive Vermeidung von Nebenkreuzungen.** Beim Torlauf kann eine
  Transitfahrt zwischen zwei Ziel-Toren zufällig über die Kante eines noch
  nicht fälligen (oder bereits erledigten) Tors führen. Das wird nicht aktiv
  verhindert (Nutzer-Entscheidung: für die Bewertung nicht relevant).
- **Kein Hilfsknoten für Parallelkanten.** Wenn zwei Kreuzungen über zwei
  verschiedene Straßen verbunden sind (z. B. direkte Verbindung + langer
  Bogen/Umweg), werden das zwei normale Kanten mit derselben Knotenpaar-
  Zuordnung — die Graph-Datenstruktur ist ein Multigraph und braucht dafür
  keinen künstlichen Zwischenknoten.
- **Zwei komplett getrennte ROS-Läufe** (Mapping vs. Torlauf), kein
  gemeinsamer Prozess. Ein Plan (`plan.json`) wird jeweils VOR dem Start
  berechnet und von den ROS-Nodes nur noch geladen.

---

## 3. Architektur-Überblick

### 3.1 Bestehende Pakete (nicht Teil dieser Challenge, nur Kontext)

- `src/packages/duckietown_msgs/` — Standard-Duckietown-Message-Definitionen.
- `src/packages/follow_lane/` — ältere, einfachere Basisversion (nur
  Lane-Following + einfache Kreuzungserkennung). Nicht weiter relevant für
  Challenge 4.
- `src/packages/explore_duckietown_ii/` — das Package, auf dem alles aufbaut.

### 3.2 Bestehende Nodes in `explore_duckietown_ii/src/` (aus einem früheren
Meilenstein, für Challenge 4 größtenteils unverändert wiederverwendet)

| Datei | Zweck |
|---|---|
| `configuration_node.py` | Tkinter-GUI zum Live-Tunen aller Parameter (JSON-Config + ROS-Topic-Broadcast) |
| `util.py` | Lädt `config/<node_name>.json`, ruft Callback bei Parameter-Updates |
| `detect_lane_node.py` | Perspektiv-Transform + HSV-Masken für weiße/gelbe Linie → Lane-Error (`/detect/lane`) |
| `detect_intersection_node.py` | Roter Streifen → `/detect/intersection` (Bool), plus Winkelmessung für Ausrichtung (`/detect/intersection_angle`) |
| `detect_apriltag_node.py` | `pupil_apriltags`-Detektor, publiziert größten erkannten Tag auf `/detect/apriltag` (Int32, alle IDs gemischt). Hat noch eine "Peak"-Tracking-Logik (`saved_apriltag`, `intersection_approaching`) aus der Kreuzungstyp-Erkennung — **wird von `switch_control_node.py` nicht mehr konsumiert** (siehe unten), läuft aber noch mit (totes Gewicht, kein Fehler). |
| `detect_duckies_node.py` | ONNX/YOLO Entenerkennung + Ausweich-Lückensuche — nicht Teil von Challenge 4, läuft nur mit. |
| `debug_view_node.py` | Kompositbild aus Debug-Infos alter Nodes (`/debug/full_view`) |
| `control_lane_node.py` | PID-Regler, führt Abbiegungen aus. **Unverändert** — bekommt nur eine Richtungs-Zahl über `/switch/intersection_direction`, egal woher sie kommt. |

### 3.3 Neue Dateien für Challenge 4 (`explore_duckietown_ii/src/`)

**Reine Python-Logik, absichtlich OHNE `rospy`-Import** (dadurch lokal ohne
ROS testbar — wichtig für den bisherigen Entwicklungsprozess):

| Datei | Zweck |
|---|---|
| `graph.py` | Kern-Datenstruktur: `Node`, `Edge`, `Hop`, `Plan`, `Graph`-Klasse (Aufbau, JSON-Persistenz, `hop_from_edge`, `edge_by_tag`, `set_gate_tag`), `Direction`-Enum + `relative_direction()` (Ein-/Ausfahrt → Links/Geradeaus/Rechts), `PlanState` (Laufzeit-Fortschritt), `save_plan`/`load_plan`. |
| `known_map.py` | Codiert die **bekannte** Stadt-Topologie hart über `Graph.add_edge()`-Aufrufe, speichert nach `config/known_map.json`. **Das ist die einzige Datei, die am Prüfungstag durch die echte Stadt ersetzt werden muss** (aktuell: Beispielstadt aus dem Foliensatz mit 3 Knoten A/B/C). |
| `mapping_planner.py` | `build_mapping_plan()`: plant für den Mapping-Lauf eine Route, die jede Kante mindestens einmal befährt (DFS mit Backtracking über bereits bekannte kürzeste Wege). Enthält auch `shortest_path_edges()` (BFS), von `gate_planner.py` wiederverwendet. |
| `gate_planner.py` | `build_gate_plan()`: übersetzt eine vorgegebene Tor-Tag-Reihenfolge in eine Route (BFS zum jeweils näheren Ende der nächsten Ziel-Kante, Zufallsentscheid bei Gleichstand). Gibt `(plan, target_hop_indices)` zurück — Letzteres markiert die *beabsichtigten* Tor-Durchfahrten (wichtig, weil Transitfahrten zufällig über andere Tor-Kanten führen können, siehe `incidental_gate_crossings()`). |
| `gate_tag_voting.py` | `pick_confident_tag(counts, min_sightings)`: Mehrheitsentscheid aus mehreren Tag-Sichtungen, verwirft bei zu wenigen Sichtungen (Schutz vor Fehldetektionen). |
| `graph_layout.py` | `compute_layout()`: schematisches Force-Directed-Layout (Fruchterman-Reingold-artig, nur numpy) für die 2D-Darstellung im Dashboard. Bewusst NICHT kompass-basiert (Parallelkanten würden das inkonsistent machen). |
| `dashboard_render.py` | `build_dashboard_image()`: reine Zeichenfunktion (Karte + Pfad-Hervorhebung + aktuelle Position + Gate-Tag-Labels + Info-Panel), gibt ein fertiges Bild zurück. |
| `compute_mapping_plan.py` | CLI-Skript (kein ROS-Node): lädt `known_map.json`, ruft `mapping_planner`, speichert `plan.json`. |
| `compute_gate_plan.py` | CLI-Skript: lädt gemappten Graphen, prüft ob alle Ziel-Tags zugeordnet sind (sonst Fehlerabbruch), ruft `gate_planner`, speichert `plan.json`. |

**ROS-Nodes:**

| Datei | Zweck |
|---|---|
| `mapping_recorder_node.py` | Läuft **nur beim Mapping-Lauf**. Verfolgt über `/switch/current_edge`, auf welcher Kante der Bot gerade fährt, zählt dabei erkannte Tor-Tags (5–13) aus `/detect/apriltag` (nur im Lane-Modus, damit Kreuzungs-Tags 1–4 nicht mitgezählt werden), ordnet sie per `pick_confident_tag()` zu. Bei Planende (`ControlType.Stop`) wird der vervollständigte Graph als `config/mapped_map.json` gespeichert. |
| `dashboard_node.py` | Dünner ROS-Wrapper: lädt Graph + Plan + Layout beim Start, abonniert `/switch/current_step`, ruft `build_dashboard_image()` auf, publiziert auf `/debug/dashboard` (2 Hz). Optional `cv2`-Fenster wenn `DUCKIE_GUI=1` gesetzt ist. |

### 3.4 Geänderte bestehende Datei

**`switch_control_node.py`** — der zentrale State-Machine-Node, angepasst für
planbasierte statt zufällige Steuerung:

- Lädt beim Start einen Plan über ROS-Parameter `~plan_path`
  (Default `config/plan.json`) als `PlanState`.
- **Entfernt:** `cbChooseDirection` (wählte früher zufällig eine Richtung
  anhand des Kreuzungstyp-Tags — Werte waren: Tag 1=┼ alle 3 Richtungen,
  Tag 3=┤ Links/Geradeaus, Tag 4=├ Geradeaus/Rechts, Tag 2=┴ Links/Rechts),
  die zugehörige `saved_apriltag`-Subscription, `import random`.
- **`cbIntersection`**: Richtung kommt jetzt aus `plan_state.next_direction()`
  statt Zufall. Erkennt zusätzlich, wenn der letzte Plan-Hop erreicht ist, und
  geht dann direkt in `ControlType.Stop` statt eine nicht existierende weitere
  Richtung abzufragen.
- **`cbIntersectionFinished`**: ruft `plan_state.advance()` auf.
- **Veröffentlicht neu:**
  - `/switch/current_edge` (Int32, Kanten-ID oder -1) — für den Recorder.
  - `/switch/current_step` (Int32, roher `PlanState.step`-Index) — fürs
    Dashboard. **Wichtig:** getrennt von `current_edge`, weil dieselbe Kante
    im Plan mehrfach vorkommen kann (Backtracking beim Mapping,
    Nebenkreuzungen beim Torlauf) — der Schritt wäre aus der Kanten-ID allein
    nicht eindeutig rekonstruierbar.
  - Beide Topics sind `latch=True`, damit später startende Subscriber sofort
    den aktuellen Stand bekommen.

### 3.5 Launch-Datei und Skripte

**`launch/explore.launch`** — neue Argumente:
- `run_mode` (`mapping`/`gate`, Default `gate` — bewusst so, damit ein
  versehentlicher Start ohne Angabe kein bestehendes `mapped_map.json`
  überschreibt)
- `plan_path` (Default `config/plan.json`)
- `map_path` (Default `config/known_map.json`)
- `run_label` (freier Text, nur fürs Dashboard-Panel)

`mapping_recorder_node` startet nur bei `run_mode:=mapping`
(`if="$(eval run_mode == 'mapping')"`). `dashboard_node` läuft in beiden
Modi mit.

**`launchers/mapping_run.sh <start_node> <start_exit>`**
Berechnet den Mapping-Plan (`compute_mapping_plan.py` gegen `known_map.json`),
startet dann `roslaunch explore_duckietown_ii explore.launch run_mode:=mapping
map_path:=known_map.json run_label:=Mapping`.

**`launchers/gate_run.sh <start_node> <start_exit> <tag1,tag2,...>`**
Berechnet den Torlauf-Plan (`compute_gate_plan.py` gegen `mapped_map.json`,
bricht mit Fehlermeldung ab falls ein Tag fehlt), startet dann `roslaunch
explore_duckietown_ii explore.launch run_mode:=gate map_path:=mapped_map.json
run_label:=Torlauf`.

---

## 4. Wie man den Bot damit fährt

### 4.1 Voraussetzung: Umgebung

Wie gehabt über `duckie-env.sh` (bzw. den `dt`/`dt-launch-sim`/`dt-pick`-Workflow
aus `Duckie_Befehle.md`): `VEHICLE_NAME`, `ROS_MASTER_URI`, `ROS_IP` müssen
gesetzt sein, `catkin_make` + `source devel/setup.bash` nach jeder Code-
Änderung.

Falls die echte Stadt vom Prüfungstag noch nicht eingetragen ist: zuerst
`src/packages/explore_duckietown_ii/src/known_map.py` anpassen (siehe
Docstring/Beispiel darin — `Graph.add_edge(node_a, exit_a, node_b, exit_b,
gate_tag=None)` pro physischer Straße, **ein Aufruf pro Straße**, nicht pro
Richtung) und einmal ausführen, um `config/known_map.json` neu zu erzeugen.

### 4.2 Mapping-Modus starten

```bash
launchers/mapping_run.sh A 1
```
(`A` = Start-Knoten, `1` = Start-Ausfahrt — beides frei wählbar, muss zu einer
tatsächlich existierenden Ausfahrt in `known_map.json` passen)

Das Skript:
1. Berechnet `config/plan.json` (Route, die jede Kante der bekannten Karte
   mindestens einmal abdeckt).
2. Startet `explore.launch` mit `run_mode:=mapping` — zusätzlich zu den
   normalen Nodes läuft `mapping_recorder_node` mit, der die Kamera-
   Tag-Erkennungen (5–13) den Kanten zuordnet.
3. Bot fährt den Plan ab, biegt an jeder Kreuzung planbasiert ab (nicht mehr
   zufällig).
4. Sobald der letzte Hop des Plans erreicht ist, stoppt der Bot automatisch
   (`ControlType.Stop`), und `mapping_recorder_node` schreibt
   `config/mapped_map.json` (der bekannte Graph, jetzt mit zugeordneten
   Tor-Tags).

**Kontrolle während des Laufs:** `/debug/dashboard`-Topic zeigt Karte + Plan +
aktuelle Position live an (z. B. via `rqt_image_view` oder `DUCKIE_GUI=1`
für ein `cv2`-Fenster).

### 4.3 Torlauf-Modus starten

Erst NACHDEM der Mapping-Lauf abgeschlossen ist und `config/mapped_map.json`
existiert:

```bash
launchers/gate_run.sh A 1 12,7,5
```
(`A 1` = Start wie oben, frei wählbar — muss nicht derselbe wie beim Mapping
sein; `12,7,5` = vorgegebene Tor-Reihenfolge als kommagetrennte Tag-IDs, kommt
von der Aufgabenstellung/dem vorgegebenen Ziel-Graphen)

Das Skript:
1. Prüft, ob alle angegebenen Tags in `mapped_map.json` einer Kante zugeordnet
   sind — falls nicht: Fehlerabbruch, kein Start (z. B. Mapping unvollständig).
2. Berechnet `config/plan.json` (Route durch die Ziel-Tore in genau dieser
   Reihenfolge).
3. Startet `explore.launch` mit `run_mode:=gate` — **ohne**
   `mapping_recorder_node` (Tags sind ja schon bekannt).
4. Bot fährt ab, stoppt automatisch am Ende.

### 4.4 Manuelle/tiefere Kontrolle (zum Debuggen)

Die Compute-Skripte können auch einzeln aufgerufen werden, z. B.:
```bash
rosrun explore_duckietown_ii compute_mapping_plan.py \
    --known-map <pfad>/known_map.json --start-node A --start-exit 1 \
    --output <pfad>/plan.json [--seed 0]

rosrun explore_duckietown_ii compute_gate_plan.py \
    --map <pfad>/mapped_map.json --tags 12,7,5 \
    --start-node A --start-exit 1 --output <pfad>/plan.json [--seed 0]
```
Danach `roslaunch explore_duckietown_ii explore.launch run_mode:=<mapping|gate>
plan_path:=<pfad>/plan.json map_path:=<pfad>/known_map.json_oder_mapped_map.json`
von Hand.

---

## 5. Relevante ROS-Topics (Überblick)

| Topic | Typ | Publisher | Subscriber | Bedeutung |
|---|---|---|---|---|
| `/detect/lane` | Float64 | `detect_lane_node` | `control_lane_node` | Lane-Error [-1,1] |
| `/detect/intersection` | Bool | `detect_intersection_node` | `switch_control_node` | Roter Streifen erkannt |
| `/detect/intersection_angle` | Float64 | `detect_intersection_node` | `control_lane_node` | Winkel für Ausrichtung |
| `/detect/apriltag` | Int32 | `detect_apriltag_node` | `mapping_recorder_node` | größter erkannter Tag im Frame (alle IDs) |
| `/switch/control` | Int32 (`ControlType`) | `switch_control_node` | `control_lane_node` | Lane / Obstacle / Stop / Intersection / Align |
| `/switch/intersection_direction` | Int32 (`IntersectionsDirections`) | `switch_control_node` | `control_lane_node` | Links/Geradeaus/Rechts, jetzt planbasiert |
| `/switch/intersection_finished` | Bool | `control_lane_node` | `switch_control_node` | Abbiegung ausgeführt |
| `/switch/align_finished` | Bool | `control_lane_node` | `switch_control_node` | Ausrichten fertig |
| `/switch/current_edge` | Int32 | `switch_control_node` | `mapping_recorder_node` | aktuelle Kanten-ID (oder -1) |
| `/switch/current_step` | Int32 | `switch_control_node` | `dashboard_node` | roher Plan-Fortschritts-Index |
| `/debug/dashboard` | CompressedImage | `dashboard_node` | (rqt/Betrachter) | Karte + Pfad + Position |

`ControlType`: `Lane=1, Obstacle=2, Stop=3, Intersection=4, Align=5`.
`IntersectionsDirections`/`Direction`: `Left=0, Straight=1, Right=2` (beide
Enums numerisch identisch, `graph.Direction` ist die ROS-unabhängige Variante).

---

## 6. Wichtige Code-Konzepte im Detail

### 6.1 Ausfahrt-Nummerierung → Abbiege-Richtung

`graph.relative_direction(entry_exit, target_exit)`:
```
diff = (target_exit - entry_exit) % 4
diff == 1 -> Right
diff == 2 -> Straight   (gegenueberliegende Ausfahrt)
diff == 3 -> Left
diff == 0 -> ungueltig (Umkehr, darf im Plan nicht vorkommen)
```

### 6.2 `Hop` und `Plan`

Ein `Hop` ist eine orientierte Kantendurchfahrt: `edge_id, from_node,
from_exit, to_node, to_exit`. Ein `Plan` ist einfach `List[Hop]`. Sowohl
Mapping-Plan als auch Gate-Plan sind strukturell identisch — der Ausführungs-
Code (`switch_control_node.py`) unterscheidet nicht, welcher Plan gerade läuft.

### 6.3 `PlanState`

Hält `step` (aktueller Index in der Hop-Liste) — das ist die **einzige**
"Positions"-Information im System, da keine Hardware-Lokalisierung existiert.
`next_direction()` berechnet die Richtung für die **nächste** Kreuzung aus
`plan[step].to_exit` und `plan[step+1].from_exit`. Gibt `None` zurück, wenn
kein weiterer Hop folgt (Ziel erreicht).

### 6.4 Warum `gate_planner.build_gate_plan()` zwei Werte zurückgibt

`(plan, target_hop_indices)` — `target_hop_indices` markiert, welche Hops die
**beabsichtigten** Tor-Durchfahrten sind. Grund: eine Transitfahrt zwischen
zwei Zielen kann zufällig über die Kante eines dritten (noch nicht fälligen)
Tors führen — das zählt NICHT als "Tor erreicht", auch wenn die Kante
denselben `gate_tag` trägt. `incidental_gate_crossings()` macht solche
Nebentreffer sichtbar (nur Logging, kein Eingriff).

---

## 7. Teststand (was schon geprüft wurde, was nicht)

**Getestet (reines Python, ohne ROS, auf einem Windows-Rechner ohne ROS-
Zugriff):**
- `mapping_planner.build_mapping_plan`: 400 Kombinationen (Startpunkte ×
  Seeds) auf der Beispielstadt — immer vollständige Kantenabdeckung.
- `gate_planner.build_gate_plan`: 600 Kombinationen (Tag-Permutationen ×
  Startpunkte × Seeds) — Zielreihenfolge immer korrekt eingehalten.
- `gate_tag_voting.pick_confident_tag`: Randfälle (leer, unter Schwelle, genau
  an Schwelle, dominanter Tag trotz Rauschen, mehrere knappe Kandidaten).
- `graph_layout.compute_layout`: Determinismus, keine Knoten-Kollisionen, auch
  auf größerem synthetischem Graphen.
- `dashboard_render.build_dashboard_image`: alle Rand-/Sonderfälle (kein
  Schritt, mitten im Plan, letzter Schritt, "Ziel erreicht" und darüber
  hinaus, leerer Plan) — keine Abstürze. Visuell geprüft (Bilder gerendert und
  angeschaut).
- `graph.py` (Persistenz, `PlanState`, `relative_direction`): Round-Trip-Tests,
  Beispiel-Testfahrten von Hand nachgerechnet.
- Alle ROS-Node-Dateien: nur **Syntax-Check** (`ast.parse`), da kein `rospy`
  auf dem Entwicklungsrechner verfügbar war.

**NICHT getestet (höchste Priorität für den nächsten Schritt):**
- Alles, was echtes ROS braucht: `roslaunch`-Verhalten (insbesondere die
  `$(eval run_mode == 'mapping')`-Bedingung in `explore.launch`), `rosrun`-
  Auffindbarkeit der neuen Skripte, Timing der neuen Topics
  (`current_edge`/`current_step`) in Echtzeit.
- Ob `detect_apriltag_node.py`s Kamera-ROI (ursprünglich auf Kreuzungs-Tags
  nahe der Kreuzung zugeschnitten) Tor-Tags mitten auf der Strecke zuverlässig
  erkennt — unbekannt, wie/wo die echten Tor-AprilTags real montiert sind.
- `min_sightings=3` in `gate_tag_voting.py` ist eine Schätzung, keine aus
  echten Daten kalibrierte Schwelle.
- Das komplette Zusammenspiel aller Nodes in einem echten Lauf.

**Bereits behoben:** Ausführungsrechte (`chmod +x`) auf allen Skripten/Nodes
wurden im Git-Commit korrigiert (waren durch Windows-Checkout verloren
gegangen) — ein frischer `git clone`/`pull` sollte also bereits die richtigen
Rechte mitbringen, kein manuelles `chmod +x` mehr nötig.

**Bekannte Einschränkung durch die Windows-Entwicklungsumgebung:**
`src/CMakeLists.txt` war im Original vermutlich ein Symlink (catkin-Konvention,
zeigt normalerweise auf `toplevel.cmake` der ROS-Distribution) und wurde durch
Git auf Windows (`core.symlinks=false`) zu einer normalen Datei mit demselben
Inhalt. Inhaltlich sollte das für `catkin_make` funktionieren (der Pfad enthält
weiterhin die richtigen CMake-Befehle), falls es doch Build-Probleme gibt,
zuerst hier nachsehen.

---

## 8. Repo-Stand

- Remote: `git@github.com:FaFelF/Duckie-Bot-Felix-Tobi.git`
- Alles liegt auf Branch **`Challenge-4`** (nicht `main`!) — `main` ist der
  ältere, weniger weit entwickelte Stand.
- Ordnerstruktur wurde bereinigt (keine doppelt verschachtelten Ordner mehr) —
  ein frischer Klon hat die Projektdateien direkt im Repo-Root.
