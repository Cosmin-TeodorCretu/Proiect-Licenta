import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import os

print("Se incarca modelul si datele...")

#incarcare model si mapare
model = tf.keras.models.load_model('model_emotii/model_emotii_best.h5')
with open('model_emotii/mapare_emotii.json') as f:
    mapare = json.load(f)

idx_to_emotie ={v: k for k, v in mapare.items()}
EMOTII =[idx_to_emotie[i] for i in range(len(mapare))]

#incarcare date test din FER2013
IMG_SIZE = 48
DATA_DIR = "fer2013/test"

test_datagen =keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print("Se genereaza predictiile...")
probabilitati = model.predict(test_generator, verbose=1)
predictii= np.argmax(probabilitati, axis=1)
etichete_reale = test_generator.classes

#matrice de confuzie
cm = confusion_matrix(etichete_reale, predictii)
etichete_display = [e.upper() for e in EMOTII]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',xticklabels=etichete_display,yticklabels=etichete_display,annot_kws={"size": 12})
plt.title('Matrice de Confuzie - Detectie Emotii', fontsize=16, pad=15)
plt.xlabel('Predictia Modelului',fontsize=13, labelpad=12)
plt.ylabel('Emotia Reala', fontsize=13, labelpad=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('matrice_confuzie_emotii.png',dpi=300)
plt.show()
print("Salvat: matrice_confuzie_emotii.png")

#raport detaliat per clasa
print('\nRAPORT PERFORMANTA EMOTII:')
print(classification_report(etichete_reale, predictii,target_names=etichete_display))