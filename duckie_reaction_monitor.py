#!/usr/bin/env python3
"""
Reaktions-Logger fuer die Enten-Ausweichung (NUR Logging, kein Eingriff).

Misst pro "Episode" (eine Ente taucht auf -> verschwindet):
  t_seen  = wann YOLO die Ente zum ersten Mal sieht   (/detect/duckies True)
  t_react = wann die Ausweichung anspringt             (/detect/duckie_control_active True)
  gap     = t_react - t_seen   <-- die entscheidende Zahl

Lesart:
  grosser gap  -> YOLO sieht die Ente lange vor der Reaktion = LOGIK-Problem
                  (wir reagieren zu spaet, obwohl sichtbar). Fix: frueher reagieren.
  gap ~ 0      -> Reaktion ~ sofort bei Sichtbarkeit. Wenn er trotzdem zu nah ist,
                  taucht die Ente erst spaet im Bild auf = FOV/Physik. Fix: langsamer
                  in Kurven.

Vergleiche bewusst KURVE vs. GERADE: ist der gap auf der Geraden gross, in der Kurve
~0, dann ist es FOV speziell in Kurven.

Beenden mit Ctrl-C -> druckt Zusammenfassung. Log auch in /tmp/duckie_reaction.log.
"""
import os, time, datetime
import rospy
from std_msgs.msg import Bool, Float64

VEH = os.environ.get('VEHICLE_NAME', 'tick')
LOG = '/tmp/duckie_reaction.log'

st = {
    'seen': False, 'reacted': False,
    't_seen': None, 't_react': None,
    'dist': 0.0,
    'gaps': [],          # Liste (gap, dist_bei_reaktion)
    'seen_no_react': 0,  # Ente gesehen, aber nie reagiert
    'episodes': 0,
}
f = open(LOG, 'w')

def ts():
    return datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]

def w(s):
    line = f'[{ts()}] {s}'
    print(line)
    f.write(line + '\n'); f.flush()

def cb_duckies(msg):
    now = time.time()
    if msg.data and not st['seen']:
        # neue Episode beginnt
        st['seen'] = True
        st['reacted'] = False
        st['t_seen'] = now
        st['episodes'] += 1
        w(f'--- Episode {st["episodes"]}: ENTE SICHTBAR (YOLO) ---')
    elif not msg.data and st['seen']:
        # Episode endet
        if not st['reacted']:
            st['seen_no_react'] += 1
            dur = now - st['t_seen']
            w(f'    Ente weg nach {dur:.2f}s, OHNE Reaktion (war nie im Reaktionsbereich/Korridor)')
        st['seen'] = False
        st['reacted'] = False

def cb_control(msg):
    now = time.time()
    if msg.data and st['seen'] and not st['reacted']:
        st['reacted'] = True
        st['t_react'] = now
        gap = now - st['t_seen']
        st['gaps'].append((gap, st['dist']))
        w(f'    >>> BOT REAGIERT. Luecke seit Sichtbarkeit = {gap:.2f}s  '
          f'(distance_factor jetzt {st["dist"]:.2f}; 1.0 = ganz nah)')

def cb_dist(msg):
    st['dist'] = msg.data

def summary():
    print('\n' + '=' * 60)
    print('ZUSAMMENFASSUNG')
    print('=' * 60)
    print(f'Episoden (Ente sichtbar): {st["episodes"]}')
    print(f'davon OHNE Reaktion:      {st["seen_no_react"]}')
    if st['gaps']:
        gs = [g for g, _ in st['gaps']]
        gs_sorted = sorted(gs)
        med = gs_sorted[len(gs_sorted) // 2]
        print(f'Reaktionen: {len(gs)}')
        print(f'Luecke YOLO->Reaktion  min/median/max = '
              f'{min(gs):.2f} / {med:.2f} / {max(gs):.2f} s')
        print('Einzelwerte (gap s, distance_factor bei Reaktion):')
        for g, d in st['gaps']:
            print(f'   gap {g:.2f}s   dist {d:.2f}')
    print('=' * 60)
    print('Deutung: grosse Luecke -> LOGIK (zu spaet trotz Sichtbarkeit).')
    print('         Luecke ~0      -> FOV/Physik (Ente erst spaet im Bild).')

if __name__ == '__main__':
    rospy.init_node('duckie_reaction_monitor', anonymous=True, disable_signals=True)
    rospy.Subscriber(f'/{VEH}/detect/duckies', Bool, cb_duckies, queue_size=50)
    rospy.Subscriber(f'/{VEH}/detect/duckie_control_active', Bool, cb_control, queue_size=50)
    rospy.Subscriber(f'/{VEH}/detect/duckie_distance_factor', Float64, cb_dist, queue_size=50)
    w(f'Monitor laeuft (Fahrzeug={VEH}). Jetzt Testdurchlaeufe fahren. Ctrl-C zum Beenden.')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        summary()
        f.close()
