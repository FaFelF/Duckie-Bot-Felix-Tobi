# Challenge 4 — Mapping & Torlauf: Aufgabe und Implementierungsleitfaden

Dieses Dokument beschreibt die Aufgabe **und** einen empfohlenen Weg, sie von
Grund auf zu implementieren. Gedacht für jemanden (oder eine Claude-Instanz),
der/die eine **eigene** Lösung baut. Am Ende stehen die Fallstricke, die uns beim
Bauen real Zeit gekostet haben — die sind das eigentlich Wertvolle, denn sie
stehen in keiner Aufgabenstellung.

Eine fertige Referenz-Implementierung liegt im selben Repo auf Branch
`Challenge-4` (`graph.py`, `mapping_planner.py`, `gate_planner.py`,
`dashboard_render.py`, `CHALLENGE4.md`), falls man vergleichen will.

---

## 1. Die Aufgabe

Ein Duckiebot fährt in einer Stadt, deren **Topologie als Graph bekannt ist**
(Kreuzungen = Knoten, Straßen = Kanten). Unbekannt ist nur, **welches „Tor"
(farbiger AprilTag, IDs 5–13) auf welcher Kante sitzt**. Zwei getrennte Läufe:

1. **Mapping-Lauf:** Der Bot fährt eine Route, die **jede Kante mindestens
   einmal** befährt, erkennt dabei die Tor-Tags und ordnet sie den Kanten zu.
   Ergebnis: die vervollständigte Karte.
2. **Torlauf (zeitkritisch):** Mit der gemappten Karte wird eine **vorgegebene
   Tor-Reihenfolge** abgefahren. Der Pfad muss **algorithmisch** berechnet
   werden, nicht hartkodiert.

Zusätzlich gefordert: ein **Dashboard**, das Karte, geplanten Pfad und aktuelle
Position zeigt.

---

## 2. Rahmenbedingungen (die falsche Annahmen verhindern)

- **Kein SLAM.** Die Topologie ist vorher bekannt — es muss nichts entdeckt
  werden, nur eine vorab geplante Route abgefahren und dabei Tags eingesammelt.
- **Keine Lokalisierung/Odometrie.** Die „Position" ist rein symbolisch: ein
  **Zähler, der bei jeder erkannten Kreuzung um 1 steigt** = Index in einer
  vorab berechneten Hop-Liste.
- **Graph ist ungerichtet** (Spur + Gegenspur = dieselbe Kante, gleicher Tag).
  Datenstruktur = **Multigraph** (zwei Kreuzungen können über mehrere Straßen
  verbunden sein — einfach mehrere Kanten, kein Hilfsknoten nötig).
- **Ausfahrt-Nummerierung ist fix und an jeder Kreuzung gleich** (kompass-
  gebunden): 1 und 3 liegen sich gegenüber, 2 rechts von 1, 4 links von 1.
  Daraus ergibt sich die Abbiegerichtung.
- **Nur der Torlauf ist zeitkritisch** — Pfad muss nicht global optimal sein,
  nur „nicht unnötig lang". → BFS (kürzeste Kreuzungsanzahl) reicht, kein
  Dijkstra/A* nötig; bei mehreren gleich kurzen Optionen zufällig wählen.
- **Startpunkt frei wählbar**, kann zwischen den zwei Läufen unterschiedlich
  sein.
- **AprilTags 1–4** bleiben für Kreuzungstypen reserviert, **5–13** sind die
  Tore.

---

## 3. Empfohlene Architektur

Trenne **reine Logik** (ohne ROS, lokal testbar) von **ROS-Nodes**.

### Reine Python-Logik (kein `rospy`)

- **`graph.py`** — Kern:
  - `Node`, `Edge`, `Hop` (orientierte Kantendurchfahrt: `edge_id, from_node,
    from_exit, to_node, to_exit`), `Plan = List[Hop]`.
  - `Graph` (Aufbau, JSON laden/speichern, `edge_by_tag`, `set_gate_tag`,
    `hop_from_edge`).
  - `relative_direction(entry_exit, target_exit) = (target - entry) % 4`
    → `1=rechts, 2=geradeaus, 3=links, 0=Wende (verboten)`.
  - `PlanState` — hält `step`, liefert `next_direction()` (Richtung an der
    nächsten Kreuzung aus `plan[step].to_exit` und `plan[step+1].from_exit`).
- **`mapping_planner.py`** — plant eine Route, die **jede Kante abdeckt**
  (DFS mit Backtracking; für Rückwege eine BFS-Wegsuche).
- **`gate_planner.py`** — übersetzt eine Tor-Reihenfolge in eine Route (zum
  jeweils näheren Ende der Ziel-Kante fahren, dann die Kante durchfahren). Gibt
  zusätzlich die Indizes der **beabsichtigten** Tor-Durchfahrten zurück (weil
  man unterwegs zufällig durch andere Tore fährt).
- **`gate_tag_voting.py`** — `pick_confident_tag`: **Mehrheitsentscheid** über
  mehrere Sichtungen pro Kante, mit Mindest-Sichtungszahl (gegen
  Fehldetektionen). **Nicht** einfach den letzten Tag nehmen.
- **`dashboard_render.py`** — reine Zeichenfunktion (gibt ein Bild zurück,
  ROS-frei).

### ROS-Nodes

- **`mapping_recorder_node`** — läuft nur beim Mapping. Verfolgt über ein Topic
  die aktuell befahrene Kante, zählt darauf die Tags 5–13 (nur im Lane-Modus,
  damit Kreuzungs-Tags 1–4 nicht mitzählen), ordnet am Kantenende per Voting zu,
  speichert am Planende die fertige Karte.
- **`dashboard_node`** — lädt Graph+Plan, abonniert den Fortschritts-Index,
  rendert das Dashboard.
- **Bestehender State-Machine-Node** (vermutlich schon da für Kreuzungen) wird
  angepasst: statt Richtung zufällig aus dem Kreuzungstyp-Tag zu wählen, liest
  er einen **vorab berechneten Plan** und gibt an jeder Kreuzung
  `next_direction()` aus. Zählt den Schritt bei jeder abgeschlossenen Kreuzung
  hoch.

### CLI-Skripte (kein ROS)

`compute_mapping_plan.py` und `compute_gate_plan.py` berechnen `plan.json`
**vor** dem Start. Zwei getrennte Läufe, kein gemeinsamer Prozess. Der
ROS-Node lädt den Plan nur noch.

---

## 4. Implementierungsschritte

1. **`graph.py`** zuerst — Datenmodell + `relative_direction` + JSON-Persistenz.
   Lokal testen (Round-Trip laden/speichern).
2. **Bekannte Karte** als Code, der `known_map.json` erzeugt (am Prüfungstag die
   echte Topologie eintragen: ein Aufruf pro Straße, nicht pro Richtung).
3. **`mapping_planner`** — volle Kantenabdeckung. Test: über viele
   Seeds/Startpunkte prüfen, dass wirklich jede Kante vorkommt.
4. **`gate_planner`** — Test: über alle Tag-Permutationen prüfen, dass die
   Zielreihenfolge exakt eingehalten wird.
5. **State-Machine-Node** auf planbasiert umbauen: Plan laden,
   `next_direction()` statt Zufall, Schritt bei „Kreuzung fertig" erhöhen, am
   letzten Hop stoppen. Zwei Topics publizieren: aktuelle Kante (für den
   Recorder) und Fortschritts-Index (fürs Dashboard).
6. **`mapping_recorder_node`** — Tags pro Kante sammeln, am Kantenende voten, am
   Ende speichern.
7. **Dashboard** — Karte + Pfad + Position rendern.
8. **Launch + Startskripte** mit `run_mode` (mapping/gate).

---

## 5. Fallstricke, die uns real Zeit gekostet haben (unbedingt beachten)

- **Wenden (U-Turns) verbieten — und zwar in der Wegsuche selbst.** Der Bot kann
  nur links/geradeaus/rechts. Ein Plan, der an einer Kreuzung durch die Einfahrt
  wieder rausgeht, lässt den Bot dort **hängen**. Es reicht **nicht**, nur den
  direkten Rückschritt zu verbieten: Die Wegsuche muss über **Zustände
  `(Knoten, Einfahrt)`** laufen statt nur über Knoten — sonst findet BFS den
  kurzen Weg, der in eine Wende läuft, und übersieht den etwas längeren
  fahrbaren. Dazu ein `validate_plan`, das einen Plan mit Wende gar nicht erst
  speichert.
- **Startkante kann schon das erste Tor tragen.** Wenn der Start-Hop bereits
  durchs erste Ziel-Tor fährt, als erledigt zählen — sonst plant er eine
  überflüssige Schleife, um dieselbe Kante nochmal zu nehmen.
- **AprilTags nur bei `hamming == 0` vertrauen.** `hamming > 0` heißt korrigierte
  Bitfehler → verwechselte IDs (z. B. 8 statt 10). Ungefilterte Lesungen
  verderben das Voting.
- **Erkennungsliste atomar setzen**, nicht „leeren, dann während der Inferenz
  neu füllen" — sonst liest ein Debug-Thread eine leere Liste
  (flackernde/fehlende Anzeige).
- **Dashboard sichtbar machen** — nicht hinter einem GUI-Flag verstecken, das aus
  X11-Gründen eh aus ist.
- **Recorder muss live publishen**, wenn das Dashboard den Mapping-Fortschritt
  zeigen soll — hält der Recorder seinen Graphen nur intern und schreibt erst am
  Ende, sieht das Dashboard nie etwas.
- **Fortschritts-Index getrennt von der Kanten-ID** publizieren: dieselbe Kante
  kommt im Plan mehrfach vor (Backtracking), die Kanten-ID allein ist nicht
  eindeutig.
- **Startpunkt validieren:** nicht jede Kreuzung hat alle vier Ausfahrten
  (T-Kreuzungen). Eine nicht existierende Ausfahrt sauber abfangen, nicht
  crashen.

---

## 6. Offene Punkte / noch nicht auf echter Hardware verifiziert

- Ob die AprilTag-Erkennung Tor-Tags **mitten auf der Strecke** zuverlässig
  liest (Kamera-ROI war ursprünglich auf Kreuzungs-Tags zugeschnitten) — hängt
  davon ab, wie/wo die echten Tore montiert sind.
- Die Mindest-Sichtungszahl fürs Voting ist eine Schätzung, keine aus echten
  Daten kalibrierte Schwelle.
- Das komplette Zusammenspiel aller Nodes in einem echten Lauf.
