import cv2
import numpy as np
import tensorflow as tf
import json


class ClasificatorEmotii:

    EMOTII_RO = {
        'angry': 'Furie', 'disgust': 'Dezgust', 'fear': 'Frica',
        'happy': 'Bucurie', 'neutral': 'Neutru', 'sad': 'Tristete',
        'surprise': 'Surpriza',
    }

    def __init__(self, calea_model="modele/model_emotii_best.h5", calea_mapare="modele/mapare_emotii.json"):
        self.model = tf.keras.models.load_model(calea_model)
        with open(calea_mapare) as f:
            self.mapare = json.load(f)

        self.idx_la_emotie = {int(v): k for k, v in self.mapare.items()}

        self.stari_clinice = {
            'neutral': ('Stable', (255, 200, 0), '#58a6ff'),
            'surprise': ('Stable', (255, 200, 0), '#58a6ff'),

            'happy': ('Comfort', (0, 255, 0), '#3fb950'),

            'sad': ('Discomfort/Pain', (0, 0, 255), '#f85149'),
            'angry': ('Discomfort/Pain', (0, 0, 255), '#f85149'),
            'fear': ('Discomfort', (0, 0, 255), '#f85149'),
            'disgust': ('Discomfort', (0, 0, 255), '#f85149'),
        }

        self.detectie_fata = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def predictie(self, fata_gri):
        fata_redimensionata = cv2.resize(fata_gri, (48, 48))
        fata_normalizata = fata_redimensionata / 255.0
        fata_input = fata_normalizata.reshape(1, 48, 48, 1)

        procente = self.model.predict(fata_input, verbose=0)[0]
        max_idx = int(np.argmax(procente))
        emotie_raw = str(self.idx_la_emotie.get(max_idx, "")).strip().lower()

        # stare_en, culoare_bgr, culoare_hex = self.stari_clinice.get(
        #     emotie_raw, ('Stable', (255, 200, 0), '#58a6ff')
        # )
        # incredere = float(procente[max_idx])

        # return emotie_raw, stare_en, incredere, culoare_bgr, culoare_hex
        incredere = float(procente[max_idx])

        PRAG_SIGURANTA_FURIE = 0.65 

        if emotie_raw == 'angry' and incredere < PRAG_SIGURANTA_FURIE:
            emotie_raw = 'neutral' 
            
            if 'neutral' in self.mapare:
                idx_neutru = int(self.mapare['neutral'])
                incredere = max(float(procente[idx_neutru]), 0.50)
            else:
                incredere = 0.50

        stare_en, culoare_bgr, culoare_hex = self.stari_clinice.get(
            emotie_raw, ('Stable', (255, 200, 0), '#58a6ff')
        )

        return emotie_raw, stare_en, incredere, culoare_bgr, culoare_hex

    
    def proceseaza_cadru(self, frame):
        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fete = self.detectie_fata.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        rezultate = []
        for (x, y, w, h) in fete:
            fata_gri = gri[y:y + h, x:x + w]
            emotie_raw, stare_en, incredere, culoare_bgr, culoare_hex = self.predictie(fata_gri)
            rezultate.append({
                'box': (x, y, w, h),
                'emotie_raw': emotie_raw,
                'emotie_ro': self.EMOTII_RO.get(emotie_raw, emotie_raw),
                'stare': stare_en,
                'incredere': incredere,
                'culoare_bgr': culoare_bgr,
                'culoare_hex': culoare_hex,
            })
        return rezultate
