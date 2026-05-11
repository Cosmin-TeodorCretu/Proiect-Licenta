*Detectia si Recunoasterea Gesturilor Mainii si Expresiilor Faciale*   
Proiect de Licenta | Cretu Cosmin-Teodor
Indrumator: dr. ing. Iosif Iulian Petrila

+Descriere:
Sistem inteligent de detectie in timp real care recunoaste simultan:
6 gesturi ale mainii: pumn inchis, palma deschisa, unu, doi, trei, ok
7 expresii faciale: furie, dezgust, frica, bucurie, neutru, tristete, surpriza

Sistemul foloseste MediaPipe pentru detectarea punctelor cheie ale mainii, retele neuronale CNN antrenate cu TensorFlow/Keras, si OpenCV pentru procesarea fluxului video in timp real.

+Structura Proiectului:
Proiect-Licenta/
│
├── interfata.py              # Aplicatia principala (ruleaza aceasta)
├── sistem_complet.py         # Versiune fara interfata grafica
│
├── colectare_date.py         # Script colectare date gesturi
├── antrenare_model.py        # Antrenare model gesturi
├── antrenare_emotii.py       # Antrenare model emotii (necesita FER2013)
│
├── clasificator_gesturi.py   # Test izolat gesturi
├── clasificator_emotii.py    # Test izolat emotii
│
├── test_mediapipe.py         # Test initial MediaPipe
├── test_date_gesturi.py      # Generare matrice de confuzie gesturi
│
├── hand_landmarker.task      # Model MediaPipe (descarcat automat)
│
├── model_gesturi/
│   ├── model_gesturi.h5      # Model antrenat gesturi
│   └── mapare_gesturi.json   # Mapare index -> nume gest
│
├── model_emotii/
│   ├── model_emotii_best.h5  # Model antrenat emotii
│   └── mapare_emotii.json    # Mapare index -> emotie
│
└── date_gesturi/             # Date colectate (CSV per gest)

+Rulare:
Instaleaza dependentele:
/pip install -r requirements.txt
Aplicatia principala:
/python interfata.py
Fara interfata grafica:
/python sistem_complet.py
Teste izolate:
/python clasificator_gesturi.py   # doar gesturi
/python clasificator_emotii.py    # doar emotii

+Gesturi Recunoscute:
-Pumn inchis Toate degetele stranse
-Palma deschisa Toate cele 5 degete intinse
-Unu Doar aratatorul ridicat
-Doi Aratator + mijlociu ridicate
-Trei Aratator + mijlociu + inelar ridicate
-OK Degetul mare + aratator formeaza cerc

+Emotii Recunoscute:
Furie, Dezgust, Frica, Bucurie, Neutru, Tristete, Surpriza
(antrenat pe datasetul public FER2013 — ~28.000 imagini)

+Performanta Modele:
Gesturi (date proprii)~100%
Emotii (FER2013)~65%
Nota: 65% acuratete pentru recunoasterea expresiilor faciale este comparabil cu performanta umana pe datasetul FER2013.


+Tehnologii Utilizate:
-Python 3.10 Limbaj principal
-MediaPipe 0.10.x Detectare puncte cheie mana/fata
-TensorFlow / Keras Antrenare si inferenta CNN
-OpenCV Procesare imagine si flux video
-NumPy Operatii matriciale
-Tkinter + Pillow Interfata grafica

+Reantrenare Modele (optional):
Gesturi — colectare date noi:
/python colectare_date.py
/python antrenare_model.py

+Emotii — necesita datasetul FER2013:
Descarca de la: https://www.kaggle.com/datasets/msambare/fer2013
Dezarhiveaza in folderul fer2013/
Ruleaza: python antrenare_emotii.py

+Note:
La prima rulare, hand_landmarker.task se descarca automat
Iluminarea buna imbunatateste semnificativ acuratetea detectiei
Modelul de gesturi a fost antrenat cu ambele maini