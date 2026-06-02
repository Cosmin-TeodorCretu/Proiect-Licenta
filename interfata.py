import cv2
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

print("Se incarca modelele...")

#incarcare model gesturi
model_gesturi = tf.keras.models.load_model("model_gesturi/model_gesturi.h5")
with open("model_gesturi/mapare_gesturi.json") as f:
    mapare_gesturi = json.load(f)
idx_to_gest= {v: k for k, v in mapare_gesturi.items()}

#incarcare model emotii
model_emotii = tf.keras.models.load_model("model_emotii/model_emotii_best.h5")
with open("model_emotii/mapare_emotii.json") as f:
    mapare_emotii = json.load(f)
idx_to_emotie = {v: k for k, v in mapare_emotii.items()}

#traduceri romana
emotii_ro = {
    'angry': 'Furie', 'disgusted': 'Dezgust', 'fearful': 'Frica',
    'happy': 'Bucurie', 'neutral': 'Neutru', 'sad': 'Tristete',
    'surprised': 'Surpriza', 'surprise': 'Surpriza',
    'fear': 'Frica', 'disgust': 'Dezgust'
}

#culori per emotie (hex pentru tkinter)
culori_emotii_hex = {
    'angry': '#ff4444', 'disgusted': '#ff8800', 'fearful': '#aa44ff',
    'happy': '#44ff88', 'neutral': '#aaaaaa', 'sad': '#4488ff',
    'surprised': '#ffff44', 'surprise': '#ffff44',
    'fear': '#aa44ff', 'disgust': '#ff8800'
}

#culori per emotie (bgr pentru opencv)
culori_emotii_bgr = {
    'angry': (0, 0, 255), 'disgusted': (0, 140, 255), 'fearful': (255, 0, 170),
    'happy': (0, 255, 136), 'neutral': (170, 170, 170), 'sad': (255, 136, 0),
    'surprised': (0, 255, 255), 'surprise': (0, 255, 255),
    'fear':(255, 0, 170), 'disgust': (0, 140, 255)
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
detector_maini= vision.HandLandmarker.create_from_options(options)

#initializare detector fata
face_cascade= cv2.CascadeClassifier(
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
        cv2.line(frame, points[start], points[end], (0, 220, 220), 2)
    for (x,y) in points:
        cv2.circle(frame, (x, y), 5, (255, 120, 0), -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

def normalizeaza_si_prezice_gest(hand_landmarks):
    baza_x = hand_landmarks[0].x
    baza_y = hand_landmarks[0].y
    baza_z= hand_landmarks[0].z
    coordonate_relative = []
    for lm in hand_landmarks:
        coordonate_relative.append(lm.x - baza_x)
        coordonate_relative.append(lm.y - baza_y)
        coordonate_relative.append(lm.z - baza_z)
    max_val = max(map(abs, coordonate_relative))
    if max_val == 0:
        max_val= 1.0
    coordonate_finale = [v / max_val for v in coordonate_relative]
    input_model = np.array([coordonate_finale], dtype=np.float32)
    probabilitati = model_gesturi.predict(input_model, verbose=0)[0]
    idx_maxim = np.argmax(probabilitati)
    return idx_to_gest[idx_maxim], probabilitati[idx_maxim]

def prezice_emotie(fata_gri):
    fata_input = cv2.resize(fata_gri, (48, 48)) / 255.0
    fata_input = fata_input.reshape(1, 48, 48, 1)
    probabilitati = model_emotii.predict(fata_input, verbose=0)[0]
    idx_maxim = np.argmax(probabilitati)
    return idx_to_emotie[idx_maxim], probabilitati[idx_maxim]


class Aplicatie:
    def __init__(self, root):
        self.root= root
        self.root.title("Detectie Gesturi si Emotii")
        self.root.geometry("1100x700")
        self.root.configure(bg="#0d1117")
        self.root.resizable(False, False)

        #stare
        self.activ = True
        self.gest_curent = "—"
        self.emotie_curenta = "—"
        self.confidenta_gest = 0.0
        self.confidenta_emotie = 0.0
        self.historic_gesturi = deque(maxlen=7)
        self.historic_emotii = deque(maxlen=7)
        self.fps_time = time.time()
        self.fps = 0

        self._construieste_ui()

        self.cap =cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._update()

    def _construieste_ui(self):
        #header
        header =tk.Frame(self.root, bg="#0d1117", height=60)
        header.pack(fill="x", padx=20, pady=(15, 0))

        tk.Label(header, text="DETECTIE GESTURI & EMOTII",
                 font=("Courier", 16, "bold"), bg="#0d1117", fg="#58a6ff").pack(side="left")

        self.lbl_fps = tk.Label(header, text="FPS: 0",
                                font=("Courier", 11), bg="#0d1117", fg="#484f58")
        self.lbl_fps.pack(side="right", padx=10)

        #corp principal
        corp = tk.Frame(self.root, bg="#0d1117")
        corp.pack(fill="both", expand=True, padx=20, pady=10)

        #video
        cadru_video = tk.Frame(corp, bg="#161b22", bd=0, relief="flat",
                               highlightbackground="#30363d", highlightthickness=1)
        cadru_video.pack(side="left", padx=(0, 15))

        self.lbl_video= tk.Label(cadru_video, bg="#161b22")
        self.lbl_video.pack(padx=2, pady=2)

        #panou dreapta
        panou = tk.Frame(corp, bg="#0d1117")
        panou.pack(side="left", fill="both", expand=True)

        #card gest
        self._card_gest(panou)

        #separator
        tk.Frame(panou, bg="#21262d", height=1).pack(fill="x", pady=12)

        #card emotie
        self._card_emotie(panou)

        #separator
        tk.Frame(panou, bg="#21262d", height=1).pack(fill="x", pady=12)

        #istoric
        self._card_istoric(panou)

        #buton inchidere
        tk.Button(self.root, text="⏹  Opreste aplicatia",
                  font=("Courier", 11, "bold"),
                  bg="#21262d", fg="#f85149",
                  activebackground="#2d1b1b", activeforeground="#f85149",
                  bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._inchide).pack(pady=(5, 15))

    def _card_gest(self, parinte):
        cadru = tk.Frame(parinte, bg="#161b22",
                         highlightbackground="#30363d", highlightthickness=1)
        cadru.pack(fill="x", pady=(0, 0))

        tk.Label(cadru, text="GEST DETECTAT",
                 font=("Courier", 9, "bold"), bg="#161b22", fg="#484f58").pack(anchor="w", padx=15, pady=(10, 2))

        self.lbl_gest = tk.Label(cadru, text="—",
                                 font=("Courier", 32, "bold"), bg="#161b22", fg="#58a6ff")
        self.lbl_gest.pack(anchor="w", padx=15)

        bara_frame = tk.Frame(cadru, bg="#161b22")
        bara_frame.pack(fill="x", padx=15, pady=(4, 10))

        tk.Label(bara_frame, text="Confidenta:",
                 font=("Courier", 9), bg="#161b22", fg="#484f58").pack(side="left")

        self.lbl_conf_gest = tk.Label(bara_frame, text="0%",
                                      font=("Courier", 9, "bold"), bg="#161b22", fg="#58a6ff")
        self.lbl_conf_gest.pack(side="right")

        self.bara_gest_bg = tk.Frame(cadru, bg="#21262d", height=4)
        self.bara_gest_bg.pack(fill="x", padx=15, pady=(0, 12))

        self.bara_gest = tk.Frame(self.bara_gest_bg, bg="#58a6ff", height=4)
        self.bara_gest.place(x=0, y=0, relheight=1, width=0)

    def _card_emotie(self, parinte):
        cadru = tk.Frame(parinte, bg="#161b22",
                         highlightbackground="#30363d", highlightthickness=1)
        cadru.pack(fill="x")

        tk.Label(cadru, text="EMOTIE DETECTATA",
                 font=("Courier", 9, "bold"), bg="#161b22", fg="#484f58").pack(anchor="w", padx=15, pady=(10, 2))

        self.lbl_emotie = tk.Label(cadru, text="—",
                                   font=("Courier", 32, "bold"), bg="#161b22", fg="#3fb950")
        self.lbl_emotie.pack(anchor="w", padx=15)

        bara_frame = tk.Frame(cadru, bg="#161b22")
        bara_frame.pack(fill="x", padx=15, pady=(4, 10))

        tk.Label(bara_frame, text="Confidenta:",
                 font=("Courier", 9), bg="#161b22", fg="#484f58").pack(side="left")

        self.lbl_conf_emotie = tk.Label(bara_frame, text="0%",
                                        font=("Courier", 9, "bold"), bg="#161b22", fg="#3fb950")
        self.lbl_conf_emotie.pack(side="right")

        self.bara_emotie_bg = tk.Frame(cadru, bg="#21262d", height=4)
        self.bara_emotie_bg.pack(fill="x", padx=15, pady=(0, 12))

        self.bara_emotie = tk.Frame(self.bara_emotie_bg, bg="#3fb950", height=4)
        self.bara_emotie.place(x=0, y=0, relheight=1, width=0)

    def _card_istoric(self, parinte):
        cadru = tk.Frame(parinte, bg="#161b22",
                         highlightbackground="#30363d", highlightthickness=1)
        cadru.pack(fill="x")

        tk.Label(cadru, text="ISTORIC (ultimele 7)",
                 font=("Courier", 9, "bold"), bg="#161b22", fg="#484f58").pack(anchor="w", padx=15, pady=(10, 6))

        self.lbl_istoric = tk.Label(cadru, text="—",
                                    font=("Courier", 10), bg="#161b22", fg="#8b949e",
                                    wraplength=340, justify="left")
        self.lbl_istoric.pack(anchor="w", padx=15, pady=(0, 12))

    def _update_ui(self, gest, conf_gest, emotie, conf_emotie):
        #gest
        gest_display =gest.replace("_", " ").upper() if gest != "—" else "—"
        self.lbl_gest.config(text=gest_display)
        self.lbl_conf_gest.config(text=f"{conf_gest*100:.0f}%")

        latime_totala =self.bara_gest_bg.winfo_width()
        self.bara_gest.place(width=int(conf_gest * latime_totala))

        #emotie
        emotie_display = emotii_ro.get(emotie, emotie).upper() if emotie != "—" else "—"
        culoare_emotie = culori_emotii_hex.get(emotie, "#3fb950")
        self.lbl_emotie.config(text=emotie_display, fg=culoare_emotie)
        self.lbl_conf_emotie.config(text=f"{conf_emotie*100:.0f}%", fg=culoare_emotie)
        self.bara_emotie.config(bg=culoare_emotie)

        latime_totala_e =self.bara_emotie_bg.winfo_width()
        self.bara_emotie.place(width=int(conf_emotie * latime_totala_e))

        #istoric
        istoric_text = " → ".join(
            [g.replace("_", " ") for g in list(self.historic_gesturi)[-7:]]
        ) if self.historic_gesturi else "—"
        self.lbl_istoric.config(text=f"Gesturi: {istoric_text}")

        #fps
        self.lbl_fps.config(text=f"FPS: {self.fps:.0f}")

    def _update(self):
        if not self.activ:
            return

        success, frame = self.cap.read()
        if not success:
            self.root.after(30, self._update)
            return

        frame = cv2.flip(frame, 1)
        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #calcul fps
        now = time.time()
        self.fps = 1.0 / (now - self.fps_time + 1e-9)
        self.fps_time =now

        gest_final = "—"
        conf_gest_final = 0.0
        emotie_final ="—"
        conf_emotie_final = 0.0

        #detectare maini
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        rezultate_maini= detector_maini.detect(mp_image)

        if rezultate_maini.hand_landmarks:
            for hand_landmarks in rezultate_maini.hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
                gest, conf_gest = normalizeaza_si_prezice_gest(hand_landmarks)

                if conf_gest > 0.8:
                    self.historic_gesturi.append(gest)
                    gest_stabil = max(set(self.historic_gesturi), key=self.historic_gesturi.count)
                    gest_final = gest_stabil
                    conf_gest_final = conf_gest

                    #text pe video
                    cv2.putText(frame, gest_stabil.replace("_", " ").upper(),
                               (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 166, 255), 2)

        #detectare fata
        fete = face_cascade.detectMultiScale(gri, 1.1, 5, minSize=(60, 60))
        for (x, y, w, h) in fete:
            fata_gri = gri[y:y+h, x:x+w]
            emotie, conf_emotie = prezice_emotie(fata_gri)

            self.historic_emotii.append(emotie)
            emotie_stabila= max(set(self.historic_emotii), key=self.historic_emotii.count)
            emotie_final = emotie_stabila
            conf_emotie_final = conf_emotie

            culoare_bgr = culori_emotii_bgr.get(emotie_stabila, (255, 255, 255))
            emotie_ro_text= emotii_ro.get(emotie_stabila, emotie_stabila).upper()

            cv2.rectangle(frame, (x, y), (x+w, y+h), culoare_bgr, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), culoare_bgr, -1)
            cv2.putText(frame, emotie_ro_text, (x+5, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        #afisare video in tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_redim= cv2.resize(frame_rgb, (640, 480))
        img = ImageTk.PhotoImage(Image.fromarray(frame_redim))
        self.lbl_video.imgtk = img
        self.lbl_video.configure(image=img)

        #actualizare UI
        self._update_ui(gest_final, conf_gest_final, emotie_final, conf_emotie_final)

        self.root.after(10, self._update)

    def _inchide(self):
        self.activ= False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicatie(root)
    root.mainloop()