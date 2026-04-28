import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import numpy as np

GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]
EXEMPLE_PER_GEST = 500

os.makedirs("date_gesturi", exist_ok=True)

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
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
        points.append((int(lm.x * w), int(lm.y * h)))
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 200), 2)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

def degete_ridicate(hand_landmarks):
    #ret degete sunt ridicate
    #feedback vizual
    degete = []
    varfuri = [8, 12, 16, 20]
    baze = [6, 10, 14, 18]
    for varf, baza in zip(varfuri, baze):
        degete.append(hand_landmarks[varf].y < hand_landmarks[baza].y)
    return degete

#instructiuni gesturi
instructiuni = {
    "pumn_inchis":    "Strange TOATE degetele in pumn",
    "palma_deschisa": "Deschide TOATE cele 5 degete larg",
    "unu":            "Ridica DOAR aratator, rest stranse",
    "doi":            "Ridica DOAR aratator+mijlociu",
    "trei":           "Ridica DOAR aratator+mijlociu+inelar",
    "ok":             "Formeaza cerc cu deget mare+aratator"
}

#sterge csv-uri vechi
for gest in GESTURI:
    fisier = f"date_gesturi/{gest}.csv"
    if os.path.exists(fisier):
        os.remove(fisier)
print("CSV-uri vechi sterse.")

cap = cv2.VideoCapture(0)

for index_gest, gest in enumerate(GESTURI):
    fisier_csv = f"date_gesturi/{gest}.csv"
    exemple_colectate = 0

    print(f"\n--- {gest.upper()} ---")
    print(f"Instructiune: {instructiuni[gest]}")
    print("Apasa SPATIU cand esti gata")

    #asteptam SPATIU
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"GEST {index_gest+1}/{len(GESTURI)}: {gest.upper()}",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, instructiuni[gest],
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Apasa SPATIU pentru a incepe",
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        if results.hand_landmarks:
            draw_landmarks(frame, results.hand_landmarks[0])

            # aratam care degete sunt ridicate ca feedback
            degete = degete_ridicate(results.hand_landmarks[0])
            nume_degete = ["Aratator", "Mijlociu", "Inelar", "Mic"]
            for i, (ridicat, nume) in enumerate(zip(degete, nume_degete)):
                culoare = (0, 255, 0) if ridicat else (0, 0, 255)
                status = "SUS" if ridicat else "JOS"
                cv2.putText(frame, f"{nume}: {status}",
                           (10, 160 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, culoare, 1)

        cv2.imshow("Colectare date", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    #colectare cu feedback vizual
    with open(fisier_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        while exemple_colectate < EXEMPLE_PER_GEST:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                #verificare corectitudine gest
                degete = degete_ridicate(hand_landmarks)
                n_ridicate = sum(degete)

                #validare simpla gest
                gest_valid = True
                if gest == "pumn_inchis" and n_ridicate > 0:
                    gest_valid = False
                elif gest == "palma_deschisa" and n_ridicate < 4:
                    gest_valid = False
                elif gest == "unu" and not (degete[0] and not degete[1] and not degete[2] and not degete[3]):
                    gest_valid = False
                elif gest == "doi" and not (degete[0] and degete[1] and not degete[2] and not degete[3]):
                    gest_valid = False
                elif gest == "trei" and not (degete[0] and degete[1] and degete[2] and not degete[3]):
                    gest_valid = False

                if gest_valid:
                    coordonate = []
                    for lm in hand_landmarks:
                        coordonate.extend([lm.x, lm.y, lm.z])
                    writer.writerow(coordonate)
                    exemple_colectate += 1

                draw_landmarks(frame, hand_landmarks)

                #feedback culoare: verde=valid, rosu=invalid
                culoare_status = (0, 255, 0) if gest_valid else (0, 0, 255)
                status_text = "VALID - salvez!" if gest_valid else "INVALID - corecteaza gestul!"
                cv2.putText(frame, status_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, culoare_status, 2)

                #aratam degete ridicate
                nume_degete = ["Aratator", "Mijlociu", "Inelar", "Mic"]
                for i, (ridicat, nume) in enumerate(zip(degete, nume_degete)):
                    culoare = (0, 255, 0) if ridicat else (0, 0, 255)
                    cv2.putText(frame, f"{nume}: {'SUS' if ridicat else 'JOS'}",
                               (10, 160 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, culoare, 1)

            #progress bar
            progress = int((exemple_colectate / EXEMPLE_PER_GEST) * 400)
            cv2.rectangle(frame, (10, 450), (410, 472), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 450), (10 + progress, 472), (0, 255, 100), -1)
            cv2.putText(frame, f"{gest.upper()}: {exemple_colectate}/{EXEMPLE_PER_GEST}",
                       (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Colectare date", frame)
            cv2.waitKey(1)

        print(f"✓ {gest}: {exemple_colectate} exemple salvate")

cap.release()
cv2.destroyAllWindows()
print("\nColectare finalizata!")