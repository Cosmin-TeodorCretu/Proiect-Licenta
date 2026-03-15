import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import json

#incarcare
model = tf.keras.models.load_model("model_gesturi/model_gesturi.h5")

with open("model_gesturi/mapare_gesturi.json") as f:
    mapare = json.load(f)

#inversare maparea: 0->gest, 1->gest ...
idx_to_gest = {v: k for k, v in mapare.items()}
print("Gesturi disponibile:", list(mapare.keys()))

#init
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 200), 2)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

def prezice_gest(hand_landmarks):
    #extragere 63 coord 
    coordonate = []
    for lm in hand_landmarks:
        coordonate.extend([lm.x, lm.y, lm.z])
    input_model = np.array([coordonate], dtype=np.float32)
    
    #obtinere prob. gest
    probabilitati = model.predict(input_model, verbose=0)[0]
    
    #luam gestul cu probabilitatea cea mai mare
    idx_maxim = np.argmax(probabilitati)
    gest = idx_to_gest[idx_maxim]
    confidenta = probabilitati[idx_maxim]
    
    return gest, confidenta, probabilitati

#bucla principala
cap = cv2.VideoCapture(0)
print("Sistem pornit. Apasa 'q' pentru a iesi.")

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        draw_landmarks(frame, hand_landmarks)

        gest, confidenta, probabilitati = prezice_gest(hand_landmarks)

        #afisare doar daca >80%
        if confidenta > 0.8:
            culoare = (0, 255, 100)
            cv2.putText(frame, f"{gest.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, culoare, 3)
            cv2.putText(frame, f"Confidenta: {confidenta*100:.1f}%", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, culoare, 2)
        else:
            cv2.putText(frame, "nesigur...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)

        #bara probabilitate
        for i, (prob, gest_nume) in enumerate(zip(probabilitati, mapare.keys())):
            y_pos = 150 + i * 35
            latime_bara = int(prob * 200)
            cv2.rectangle(frame, (10, y_pos), (210, y_pos + 22), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, y_pos), (10 + latime_bara, y_pos + 22), (0, 200, 200), -1)
            cv2.putText(frame, f"{gest_nume}: {prob*100:.0f}%", (215, y_pos + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Nicio mana detectata", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    cv2.imshow("Clasificator Gesturi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()