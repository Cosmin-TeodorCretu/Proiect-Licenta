import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ClasificatorGesturi:
    CONEXIUNI_PALMA = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
    ]

    GEST_URGENTA = 'pumn_inchis'

    def __init__(self, calea_model="modele/model_gesturi.h5", calea_mapare="modele/mapare_gesturi.json",
                 calea_task="modele/hand_landmarker.task"):
        self.model = tf.keras.models.load_model(calea_model)
        with open(calea_mapare) as f:
            self.mapare = json.load(f)

        self.idx_la_gest = {int(v): k for k, v in self.mapare.items()}

        self.comenzi_medicale = {
            'pumn_inchis': 'MEDICAL EMERGENCY!',
            'ok': 'I need water.',
            'unu': 'I am in pain. \nNeed painkillers.',
            'doi': 'Please call a nurse.',
            'trei': 'Please adjust the bed.',
            'palma_deschisa': 'Cancel command.'
        }

        base_options = python.BaseOptions(model_asset_path=calea_task)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def deseneaza_palma(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        puncte = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start, end in self.CONEXIUNI_PALMA:
            cv2.line(frame, puncte[start], puncte[end], (0, 215, 255), 2, cv2.LINE_AA)
        for (x, y) in puncte:
            cv2.circle(frame, (x, y), 5, (255, 100, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    def predictie(self, hand_landmarks):
        base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
        coordonate_relative = []

        for lm in hand_landmarks:
            coordonate_relative.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

        val_max = max(map(abs, coordonate_relative)) or 1.0
        coordonate_finale = [v / val_max for v in coordonate_relative]

        input_model = np.array([coordonate_finale], dtype=np.float32)
        procente = self.model.predict(input_model, verbose=0)[0]
        max_idx = int(np.argmax(procente))

        gest_raw = self.idx_la_gest.get(max_idx, "").strip().lower()
        comanda_en = self.comenzi_medicale.get(gest_raw, "Unknown Command")

        return gest_raw, comanda_en, float(procente[max_idx])

    def proceseaza_cadru(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        rezultat_detectie = self.detector.detect(mp_image)

        if rezultat_detectie.hand_landmarks:
            landmarks = rezultat_detectie.hand_landmarks[0]
            gest_raw, comanda_en, incredere = self.predictie(landmarks)
            return landmarks, gest_raw, comanda_en, incredere
        return None, None, None, 0.0
