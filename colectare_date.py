import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

#config
GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]
EXEMPLE_PER_GEST = 300
#GESTURI = ["palma_deschisa", "doi", "trei"]
#EXEMPLE_PER_GEST = 600
model_path = "hand_landmarker.task"

#folder date
os.makedirs("date_gesturi", exist_ok=True)

#init
base_options = python.BaseOptions(model_asset_path=model_path)
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

def extrage_coordonate(hand_landmarks):
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
        
    coordonate_finale = [val / max_val for val in coordonate_relative]
        
    return coordonate_finale

#bucla principala colectare date
cap = cv2.VideoCapture(0)

for index_gest, gest in enumerate(GESTURI):
    fisier_csv = f"date_gesturi/{gest}.csv"
    exemple_colectate = 0
    
    print(f"\nPregateste-te pentru gestul: {gest.upper()}")
    print(f"Apasa SPATIU cand esti gata sa incepi colectarea")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        
        #instructiuni
        cv2.putText(frame, f"Gest: {gest.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, "Apasa SPATIU pentru a incepe", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gest {index_gest + 1} din {len(GESTURI)}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Colectare date", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    #write CSV
    with open(fisier_csv, 'w', newline='') as f:
    #with open(fisier_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        
        #colect 200 exemplare
        while exemple_colectate < EXEMPLE_PER_GEST:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                #extragere coordonate + CSV
                coordonate = extrage_coordonate(results.hand_landmarks[0])
                writer.writerow(coordonate)
                exemple_colectate += 1
            
            #progress bar
            progress = int((exemple_colectate / EXEMPLE_PER_GEST) * 400)
            cv2.rectangle(frame, (10, 450), (410, 475), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 450), (10 + progress, 475), (0, 255, 100), -1)
            
            cv2.putText(frame, f"{gest.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)
            cv2.putText(frame, f"Colectat: {exemple_colectate}/{EXEMPLE_PER_GEST}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Tine gestul constant!", (10, 425),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
            if results.hand_landmarks:
                draw_landmarks(frame, results.hand_landmarks[0])
            
            cv2.imshow("Colectare date", frame)
            cv2.waitKey(1)
        
        print(f"✓ {gest}: {exemple_colectate} exemple salvate in {fisier_csv}")

cap.release()
cv2.destroyAllWindows()
print("\nColectare finalizata! Fisierele CSV sunt in folderul 'date_gesturi'")