import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

print("Se incarca modelele... dureaza cateva secunde.")

#incarcare modele
model_gesturi = tf.keras.models.load_model("model_gesturi/model_gesturi.h5")
with open("model_gesturi/mapare_gesturi.json") as f:
    mapare_gesturi = json.load(f)
idx_to_gest = {v: k for k, v in mapare_gesturi.items()}

model_emotii = tf.keras.models.load_model("model_emotii/model_emotii_best.h5")
with open("model_emotii/mapare_emotii.json") as f:
    mapare_emotii = json.load(f)
idx_to_emotie = {v: k for k, v in mapare_emotii.items()}

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

#dictionar culori emotii
culori_emotii_bgr = {
    'angry': (0, 0, 255),     
    'disgusted': (0, 100, 0),    
    'fearful': (128, 0, 128),   
    'fear': (128, 0, 128),     
    'happy': (0, 255, 0),      
    'neutral': (128, 128, 128),
    'sad': (255, 0, 0),        
    'surprised': (0, 255, 255),
    'surprise': (0, 255, 255)  
}

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector_maini = vision.HandLandmarker.create_from_options(options)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extrage_coordonate_mana(hand_landmarks):
    coordonate = []
    for lm in hand_landmarks:
        coordonate.extend([lm.x, lm.y, lm.z])
    return np.array([coordonate], dtype=np.float32)

def prezice_gest(hand_landmarks):
    date = extrage_coordonate_mana(hand_landmarks)
    predictii = model_gesturi.predict(date, verbose=0)[0]
    idx_max = np.argmax(predictii)
    return idx_to_gest[idx_max], predictii[idx_max]

def prezice_emotie(fata_gri):
    fata_resize = cv2.resize(fata_gri, (48, 48))
    fata_norm = fata_resize / 255.0
    fata_reshaped = np.reshape(fata_norm, (1, 48, 48, 1))
    predictii = model_emotii.predict(fata_reshaped, verbose=0)[0]
    idx_max = np.argmax(predictii)
    return idx_to_emotie[idx_max], predictii[idx_max]


#interfata grafica
class AplicatieLicenta:
    def __init__(self, fereastra):
        self.fereastra = fereastra
        self.fereastra.title("Licenta - Recunoastere Gesturi si Emotii")
        self.fereastra.geometry("900x670") # am marit putin inaltimea
        self.fereastra.configure(bg="#2c3e50") 
        
        titlu = tk.Label(fereastra, text="Interfata Detectie - Timp Real", 
                         font=("Helvetica", 20, "bold"), bg="#2c3e50", fg="white")
        titlu.pack(pady=10)
        
        self.video_label = tk.Label(fereastra)
        self.video_label.pack()
        
        buton_exit = tk.Button(fereastra, text="Inchide Aplicatia", font=("Helvetica", 14), 
                               bg="#e74c3c", fg="white", command=self.inchide_aplicatia)
        buton_exit.pack(pady=20)
        
        #init camera+memorie
        self.cap = cv2.VideoCapture(0)
        self.istoric_gest = deque(maxlen=5)
        self.istoric_emotie = deque(maxlen=5)
        
        self.update_frame()

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            h_frame, w_frame, _ = frame.shape
            
            #procesare maini
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            rezultate_maini = detector_maini.detect(mp_image)
            
            if rezultate_maini.hand_landmarks:
                punct_baza = rezultate_maini.hand_landmarks[0][0]
                cx, cy = int(punct_baza.x * w_frame), int(punct_baza.y * h_frame)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                
                gest, conf_gest = prezice_gest(rezultate_maini.hand_landmarks[0])
                self.istoric_gest.append(gest)
                gest_stabil = max(set(self.istoric_gest), key=self.istoric_gest.count)
                
                if conf_gest > 0.8:
                    cv2.putText(frame, f"Gest: {gest_stabil.upper()}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            #procesare fata
        
            gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fete = face_cascade.detectMultiScale(gri, 1.1, 5, minSize=(48, 48))
            
            for (x, y, w, h) in fete:
                
                fata_gri = gri[y:y+h, x:x+w]
                emotie, conf_emotie = prezice_emotie(fata_gri)
                
                self.istoric_emotie.append(emotie)
                emotie_stabila = max(set(self.istoric_emotie), key=self.istoric_emotie.count)
                
                culoare_care_se_schimba = culori_emotii_bgr.get(emotie_stabila, (255, 255, 255))
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), culoare_care_se_schimba, 2)
                
                emotie_tradusa = emotii_ro.get(emotie_stabila, emotie_stabila)
                cv2.putText(frame, f"Emotie: {emotie_tradusa.upper()}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, culoare_care_se_schimba, 2)
            
            #afisare Tkinter
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)
            
        self.fereastra.after(10, self.update_frame)

    def inchide_aplicatia(self):
        print("Inchidem camera si aplicatia...")
        self.cap.release()
        self.fereastra.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicatieLicenta(root)
    root.mainloop()