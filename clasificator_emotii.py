import cv2
import numpy as np
import tensorflow as tf
import json
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

#incarcare model si mapare
model = tf.keras.models.load_model("model_emotii/model_emotii_best.h5")

with open("model_emotii/mapare_emotii.json") as f:
    mapare = json.load(f)

#inversam maparea: 0->emotie, 1->emotie etc.
idx_to_emotie = {v: k for k, v in mapare.items()}
print("Emotii disponibile:", list(mapare.keys()))

#emotii in romana pentru afisare
emotii_ro = {
    'angry': 'furie',
    'disgusted': 'dezgust',
    'fearful': 'frica',
    'happy': 'bucurie',
    'neutral': 'neutru',
    'sad': 'tristete',
    'surprised': 'surpriza',
    'surprise': 'surpriza',
    'fear': 'frica',
    'disgust': 'dezgust'
}

#culori per emotie
culori_emotii = {
    'angry':     (0, 0, 255),
    'disgusted': (0, 140, 255),
    'fearful':   (255, 0, 255),
    'happy':     (0, 255, 0),
    'neutral':   (255, 255, 255),
    'sad':       (255, 100, 0),
    'surprised': (0, 255, 255),
    'surprise':  (0, 255, 255),
    'fear':      (255, 0, 255),
    'disgust':   (0, 140, 255)
}

#initializare detector fata (haar cascade - simplu si rapid)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def prezice_emotie(fata_gri):
    #redimensionam la 48x48 cat asteapta modelul
    fata_redim = cv2.resize(fata_gri, (48, 48))
    #normalizam pixelii din 0-255 la 0-1
    fata_norm = fata_redim / 255.0
    #adaugam dimensiunile batch si canal: (48,48) -> (1,48,48,1)
    fata_input = fata_norm.reshape(1, 48, 48, 1)

    #predictie
    probabilitati = model.predict(fata_input, verbose=0)[0]
    idx_maxim = np.argmax(probabilitati)
    emotie = idx_to_emotie[idx_maxim]
    confidenta = probabilitati[idx_maxim]

    return emotie, confidenta, probabilitati

#bucla principala
cap = cv2.VideoCapture(0)
print("Sistem pornit. Apasa 'q' pentru a iesi.")

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    #convertim in gri pentru detectarea fetei
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectam fetele in imagine
    fete = face_cascade.detectMultiScale(
        gri,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    for (x, y, w, h) in fete:
        #extragem doar fata din imagine
        fata_gri = gri[y:y+h, x:x+w]

        emotie, confidenta, probabilitati = prezice_emotie(fata_gri)

        culoare = culori_emotii.get(emotie, (255, 255, 255))
        emotie_ro = emotii_ro.get(emotie, emotie)

        #dreptunghi in jurul fetei
        cv2.rectangle(frame, (x, y), (x+w, y+h), culoare, 2)

        #eticheta deasupra dreptunghiului
        eticheta = f"{emotie_ro.upper()} {confidenta*100:.0f}%"
        cv2.rectangle(frame, (x, y-35), (x+w, y), culoare, -1)
        cv2.putText(frame, eticheta, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        #bara de probabilitati pentru fiecare emotie
        for i, (emotie_nume, idx) in enumerate(mapare.items()):
            prob = probabilitati[idx]
            y_pos = 30 + i * 30
            latime_bara = int(prob * 150)
            cv2.rectangle(frame, (10, y_pos), (160, y_pos+20), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, y_pos), (10+latime_bara, y_pos+20),
                         culori_emotii.get(emotie_nume, (200,200,200)), -1)
            emotie_ro_bara = emotii_ro.get(emotie_nume, emotie_nume)
            cv2.putText(frame, f"{emotie_ro_bara}: {prob*100:.0f}%",
                       (165, y_pos+15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (255, 255, 255), 1)

    if len(fete) == 0:
        cv2.putText(frame, "Nicio fata detectata", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    cv2.imshow("Clasificator Emotii", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()