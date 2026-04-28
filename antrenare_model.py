import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#incarcare date
GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]

date = []
etichete = []

# for gest in GESTURI:
#     fisier = f"date_gesturi/{gest}.csv"
#     df = pd.read_csv(fisier, header=None)
    
#     for _, rand in df.iterrows():
#         date.append(rand.values)
#         etichete.append(gest)
#v1

for gest in GESTURI:
    fisier = f"date_gesturi/{gest}.csv"
    df = pd.read_csv(fisier, header=None)
    
    for _, rand in df.iterrows():
        valori = rand.values
        
        date.append(valori)
        etichete.append(gest)
        
        valori_oglindite = valori.copy()
        for i in range(0, len(valori_oglindite), 3):
            valori_oglindite[i] = -valori_oglindite[i] 
        date.append(valori_oglindite)
        etichete.append(gest)
#v2

date = np.array(date, dtype=np.float32)
etichete = np.array(etichete)

print(f"Total exemple: {len(date)}")
print(f"Forma datelor: {date.shape}")  # (1200, 63)
print(f"Gesturi unice: {np.unique(etichete)}")

#codificare etichete (text -> numere)
#pumn_inchis=0,palma_deschisa=1,unu=2 ...
encoder = LabelEncoder()
etichete_numerice = encoder.fit_transform(etichete)

etichete_onehot = keras.utils.to_categorical(etichete_numerice)

print(f"\nMapare gesturi -> numere:")
for i, gest in enumerate(encoder.classes_):
    print(f"  {i} = {gest}")

#impartire date  antrenare80/test20
X_train, X_test, y_train, y_test = train_test_split(
    date, etichete_onehot,
    test_size=0.2,
    random_state=42,
    stratify=etichete_onehot
)

print(f"\nDate antrenare: {X_train.shape[0]} exemple")
print(f"Date testare:   {X_test.shape[0]} exemple")

#arhitectura retelei
model = keras.Sequential([
    #strat intrare
    keras.layers.Input(shape=(63,)),
    
    #strat1: 128 neuroni
    #relu = functia de activare (introduce neliniaritate)
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    
    #strat2: 64 neuroni
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    
    #strat3: 32 neuroni
    keras.layers.Dense(32, activation='relu'),
    
    #strat iesire: 6 neuroni
    #softmax transforma valorile in probabilitati (suma = 1)
    keras.layers.Dense(6, activation='softmax')
])

model.summary()

#compilare-
model.compile(
    #adam = algoritmul de optimizare (ajusteaza ponderile)
    optimizer='adam',
    #categorical crossentropy = functia de pierdere pentru clasificare multipla
    loss='categorical_crossentropy',
    #accuracy = metrica pe care o urmarim
    metrics=['accuracy']
)

#antranare
print("\nIncepe antrenarea...")

history = model.fit(
    X_train, y_train,
    epochs=50,           # numarul de treceri prin tot datasetul
    batch_size=32,       # cate exemple proceseaza odata
    validation_split=0.1, # 10% din datele de antrenare pentru validare
    verbose=1
)

#evaluare
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcuratete pe datele de test: {accuracy * 100:.1f}%")

#save + encoder
os.makedirs("model_gesturi", exist_ok=True)
model.save("model_gesturi/model_gesturi.h5")

#save + mapare
import json
mapare = {gest: int(i) for i, gest in enumerate(encoder.classes_)}
with open("model_gesturi/mapare_gesturi.json", "w") as f:
    json.dump(mapare, f, indent=2)

print("\nModel salvat in 'model_gesturi/model_gesturi.h5'")
print("Mapare salvata in 'model_gesturi/mapare_gesturi.json'")

#grafic acuratete
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Antrenare')
plt.plot(history.history['val_accuracy'], label='Validare')
plt.title('Acuratete')
plt.xlabel('Epoca')
plt.ylabel('Acuratete')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Validare')
plt.title('Pierdere (Loss)')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("model_gesturi/grafic_antrenare.png")
plt.show()
print("Grafic salvat in 'model_gesturi/grafic_antrenare.png'")