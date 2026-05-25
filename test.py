import onnxruntime as ort
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # kein Display nötig
from matplotlib import pyplot as plt

session = ort.InferenceSession("/tmp/duckie_v19.onnx")
input_name = session.get_inputs()[0].name
print("Modell geladen. Input:", session.get_inputs()[0].shape)

IMAGE_PATH = "/root/DuckieRace/test0_jpg.rf.A8ZUDeF890gCpX98CNEJ.jpg"
CONF_THRESHOLD = 0.3

orig = cv2.imread(IMAGE_PATH)
img = cv2.resize(orig, (640, 480))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

inp = img_rgb.astype(np.float32) / 255.0
inp = np.transpose(inp, (2, 0, 1))
inp = np.expand_dims(inp, 0)

# Inferenz
raw_output = session.run(None, {input_name: inp})
detections = raw_output[0][0]  # shape: (300, 6) – alle erkannten Boxen

print(f"Output shape: {detections.shape}")
print(f"\nErste 5 Boxen (roh):")
print(detections[:5])

# Alle Boxen mit ausreichend hoher Confidence ausgeben
print(f"\nBoxen mit conf > {CONF_THRESHOLD}:")
for i, box in enumerate(detections):
    x1, y1, x2, y2, conf, class_id = box
    if conf >= CONF_THRESHOLD:
        print(f"  Box {i}: x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f}  conf={conf:.3f}")

# Bild mit Bounding Boxes anzeigen
result_img = img_rgb.copy()
for box in detections:
    x1, y1, x2, y2, conf, class_id = box
    if conf < CONF_THRESHOLD:
        continue
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(result_img, f"Ente {conf:.2f}", (x1, max(y1-6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

plt.figure(figsize=(12, 7))
plt.imshow(result_img)
plt.axis("off")
plt.title(f"Erkannte Enten (conf > {CONF_THRESHOLD})")
plt.savefig("/root/DuckieRace/duckie_result.jpg", bbox_inches='tight')
print("Bild gespeichert: duckie_result.jpg")