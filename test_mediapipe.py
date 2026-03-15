import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

#descarcare model
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Se descarca modelul...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model descarcat!")

#configurare detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

#conexiuni punctele maini
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # degetul mare
    (0,5),(5,6),(6,7),(7,8),        # aratator
    (0,9),(9,10),(10,11),(11,12),   # mijlociu
    (0,13),(13,14),(14,15),(15,16), # inelar
    (0,17),(17,18),(18,19),(19,20), # mic
    (5,9),(9,13),(13,17)            # palma
]

def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    
    #convertim coordonatele din 0-1 in pixeli
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
    
    #linii conexiune
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 200), 2)
    
    #desenam punctele
    for i, (x, y) in enumerate(points):
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

cap = cv2.VideoCapture(0)
print("Camera pornita. Apasa 'q' pentru a iesi.")

def detecteaza_gest(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    
    #extragem puncte relevante
    #varfuri degete: 4, 8, 12, 16, 20
    #articulatii mijloc: 3, 6, 10, 14, 18
    
    deget_mare  = hand_landmarks[4].y * h
    aratator    = hand_landmarks[8].y * h
    mijlociu    = hand_landmarks[12].y * h
    inelar      = hand_landmarks[16].y * h
    mic         = hand_landmarks[20].y * h
    
    #articulatii de baza (mai jos = y mai mare)
    baza_aratator = hand_landmarks[6].y * h
    baza_mijlociu = hand_landmarks[10].y * h
    baza_inelar   = hand_landmarks[14].y * h
    baza_mic      = hand_landmarks[18].y * h
    
    # un deget e "ridicat" daca varful lui e mai sus decat baza
    # "mai sus" in imagine = y mai mic
    aratator_ridicat = aratator    < baza_aratator
    mijlociu_ridicat = mijlociu    < baza_mijlociu
    inelar_ridicat   = inelar      < baza_inelar
    mic_ridicat      = mic         < baza_mic
    
    #logica gesturi
    if aratator_ridicat and mijlociu_ridicat and inelar_ridicat and mic_ridicat:
        return "STOP"
    elif aratator_ridicat and not mijlociu_ridicat and not inelar_ridicat and not mic_ridicat:
        return "UNUL"
    elif aratator_ridicat and mijlociu_ridicat and not inelar_ridicat and not mic_ridicat:
        return "DOUA"
    elif aratator_ridicat and mijlociu_ridicat and inelar_ridicat and not mic_ridicat:
        return "TREI"
    else:
        return "necunoscut"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # 1 = flip orizontal (oglinda)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #detectam mana
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)
            
            #detectam gestul
            gest = detecteaza_gest(hand_landmarks, frame.shape)
            
            #afisam gestul pe ecran   culori diferite
            culoare = (0, 255, 0) if gest != "necunoscut" else (0, 100, 255)
            cv2.putText(frame, gest, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, culoare, 3)
        
        cv2.putText(frame, f"Maini: {len(results.hand_landmarks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    #afisam coordonatele primei maini detectate in consola
    if results.hand_landmarks:
        mana = results.hand_landmarks[0]
        print(f"Varf aratator (punct 8): x={mana[8].x:.2f}, y={mana[8].y:.2f}, z={mana[8].z:.2f}")

    cv2.imshow("Detectare mana", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()