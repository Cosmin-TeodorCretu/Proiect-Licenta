import numpy as np
import pandas as pd
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("Se incarca modelul si datele...")

#incarcare+mapare gesturi
model = tf.keras.models.load_model('model_gesturi/model_gesturi.h5')
with open('model_gesturi/mapare_gesturi.json') as f:
    mapare = json.load(f)

#sortare gesturi
idx_to_gest = {v: k for k, v in mapare.items()}
GESTURI = [idx_to_gest[i] for i in range(len(mapare))]

#incarcare date
date, etichete = [], []
for gest in GESTURI:
    try:
        df = pd.read_csv(f'date_gesturi/{gest}.csv', header=None)
        for _, rand in df.iterrows():
            date.append(rand.values)
            etichete.append(mapare[gest])
    except FileNotFoundError:
        print(f"Atentie: Fisierul pentru {gest} nu a fost gasit.")

date = np.array(date, dtype=np.float32)
etichete_adevarate = np.array(etichete)

print("Se genereaza predictiile...")
predictii = np.argmax(model.predict(date, verbose=0), axis=1)

#generare+salvare matrice de confuzie
cm = confusion_matrix(etichete_adevarate, predictii)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[g.replace('_', ' ').upper() for g in GESTURI], 
            yticklabels=[g.replace('_', ' ').upper() for g in GESTURI],
            annot_kws={"size": 14}) # Cifre mai mari in interiorul casutelor

plt.title('Matrice de Confuzie - Detectie Gesturi', fontsize=18, pad=20)
plt.xlabel('Predictia Modelului ', fontsize=14, labelpad=15)
plt.ylabel('Gestul Real (Ce era de fapt)', fontsize=14, labelpad=15)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

#salvare grafic
nume_imagine = 'matrice_confuzie_gesturi.png'
plt.savefig(nume_imagine, dpi=300)
print(f"\n=> SUCCES: Imaginea '{nume_imagine}' a fost salvata in folderul proiectului!")

#generare raport (de copiat in teza)
print('RAPORT DE PERFORMANTA PENTRU CAPITOLUL 4 (LICENTA)')
raport = classification_report(etichete_adevarate, predictii, target_names=[g.upper() for g in GESTURI])
print(raport)