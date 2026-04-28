import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

#incarcare model gesturi
model_gesturi = tf.keras.models.load_model("model_gesturi/model_gesturi.h5")
with open("model_gesturi/mapare_gesturi.json") as f:
    mapare_gesturi = json.load(f)
idx_to_gest = {v: k for k, v in mapare_gesturi.items()}

#incarcare model emotii
model_emotii = tf.keras.models.load_model("model_emotii/model_emotii_best.h5")
with open("model_emotii/mapare_emotii.json") as f:
    mapare_emotii = json.load(f)
idx_to_emotie = {v: k for k, v in mapare_emotii.items()}

#traduceri romana
emotii_ro = {
    'angry': 'furie', 'disgusted': 'dezgust', 'fearful': 'frica',
    'happy': 'bucurie', 'neutral': 'neutru', 'sad': 'tristete',
    'surprised': 'surpriza', 'surprise': 'surpriza',
    'fear': 'frica', 'disgust': 'dezgust'
}
culori_emotii = {
    'angry': (0,0,255), 'disgusted': (0,140,255), 'fearful': (255,0,255),
    'happy': (0,255,0), 'neutral': (255,255,255), 'sad': (255,100,0),
    'surprised': (0,255,255), 'surprise': (0,255,255),
    'fear': (255,0,255), 'disgust': (0,140,255)
}

#initializare detector maini
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector_maini = vision.HandLandmarker.create_from_options(options)

#initializare detector fata
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

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
        points.append((int(lm.x * w), int(lm.y * h)))
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 200), 2)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

def prezice_gest(hand_landmarks):
    baza_x = hand_landmarks[0].x
    baza_y = hand_landmarks[0].y
    baza_z = hand_landmarks[0].z

    coordonate_relative = []
    
    for lm in hand_landmarks:
        coordonate_relative.append(lm.x - baza_x)
        coordonate_relative.append(lm.y - baza_y)
        coordonate_relative.append(lm.z - baza_z)
        
    max_val = max(list(map(abs, coordonate_relative)))
    if max_val == 0:
        max_val = 1.0
        
    coordonate_finale = [v / max_val for v in coordonate_relative]
        
    input_model = np.array([coordonate_finale], dtype=np.float32)
    
    probabilitati = model_gesturi.predict(input_model, verbose=0)[0]
    idx_maxim = np.argmax(probabilitati)
    gest_predzis = idx_to_gest[idx_maxim]
    confidenta = probabilitati[idx_maxim]
    
    return gest_predzis, confidenta, probabilitati
#v3

def prezice_emotie(fata_gri):
    fata_input = cv2.resize(fata_gri, (48, 48))
    fata_input = fata_input / 255.0
    fata_input = fata_input.reshape(1, 48, 48, 1)
    probabilitati = model_emotii.predict(fata_input, verbose=0)[0]
    idx_maxim = np.argmax(probabilitati)
    return idx_to_emotie[idx_maxim], probabilitati[idx_maxim]

#bucla principala
#mem pt ult 5 cadre
istoric_gesturi = deque(maxlen=5)
istoric_emotii = deque(maxlen=5)
cap = cv2.VideoCapture(0)
print("Sistem complet pornit. Apasa 'q' pentru a iesi.")

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectare si clasificare gesturi
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results_maini = detector_maini.detect(mp_image)

    if results_maini.hand_landmarks:
        for i, hand_landmarks in enumerate(results_maini.hand_landmarks):
            draw_landmarks(frame, hand_landmarks)

            gest, confidenta_gest, _ = prezice_gest(hand_landmarks)

            istoric_gesturi.append(gest)
        
            gest_stabil = max(set(istoric_gesturi), key=istoric_gesturi.count)

            if confidenta_gest > 0.8:
                cv2.putText(frame, f"GEST: {gest_stabil.upper()}", (10, 45 + i*60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
                cv2.putText(frame, f"Conf: {confidenta_gest*100:.1f}%", (10, 75 + i*60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
            else:
                cv2.putText(frame, "GEST: nesigur", (10, 45 + i*60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    #detectare si clasificare emotii
    fete = face_cascade.detectMultiScale(gri, 1.1, 5, minSize=(48, 48))

    for (x, y, w, h) in fete:
        fata_gri = gri[y:y+h, x:x+w]
        emotie, confidenta_emotie = prezice_emotie(fata_gri)

        istoric_emotii.append(emotie)
        
        emotie_stabila = max(set(istoric_emotii), key=istoric_emotii.count)

        culoare = culori_emotii.get(emotie_stabila, (255, 255, 255))
        emotie_ro = emotii_ro.get(emotie_stabila, emotie_stabila)

        cv2.rectangle(frame, (x, y), (x+w, y+h), culoare, 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), culoare, -1)
        
        cv2.putText(frame, f"{emotie_ro.upper()} {confidenta_emotie*100:.0f}%", 
                    (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    #informatii generale
    h_frame, w_frame = frame.shape[:2]
    cv2.putText(frame, f"Maini: {len(results_maini.hand_landmarks) if results_maini.hand_landmarks else 0}",
               (w_frame-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Fete: {len(fete)}",
               (w_frame-150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sistem Complet - Gesturi si Emotii", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()