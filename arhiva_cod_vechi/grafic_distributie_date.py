import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]

#numaram exemplele per gest
nr_exemple= []
for gest in GESTURI:
    fisier = f"date_gesturi/{gest}.csv"
    df = pd.read_csv(fisier, header=None)
    nr_exemple.append(len(df))

#numaram imaginile per emotie din FER2013
EMOTII_FER = ['angry','disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprise']
emotii_ro = ['Furie', 'Dezgust', 'Frica','Bucurie', 'Neutru', 'Tristete', 'Surpriza']
nr_imagini_train = []
nr_imagini_test = []

for emotie in EMOTII_FER:
    folder_train= f"fer2013/train/{emotie}"
    folder_test = f"fer2013/test/{emotie}"
    nr_imagini_train.append(len(os.listdir(folder_train)) if os.path.exists(folder_train) else 0)
    nr_imagini_test.append(len(os.listdir(folder_test)) if os.path.exists(folder_test) else 0)

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

#grafic 1: distributie gesturi
ax1 = fig.add_subplot(gs[0, 0])
culori_gesturi = ['#58a6ff', '#3fb950', '#ff8c00', '#ff4444', '#aa44ff', '#ffff44']
bars = ax1.bar([g.replace("_", "\n") for g in GESTURI], nr_exemple,
               color=culori_gesturi, edgecolor='#30363d', linewidth=0.8)
ax1.set_title('Distributia Datelor - Gesturi', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Gest', fontsize=10)
ax1.set_ylabel('Numar exemple', fontsize=10)
ax1.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')
ax1.tick_params(colors='white')
ax1.title.set_color('white')
ax1.xaxis.label.set_color('white')
ax1.yaxis.label.set_color('white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#30363d')
for bar, val in zip(bars, nr_exemple):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             str(val), ha='center', va='bottom', fontsize=9, color='white')

#grafic 2: distributie emotii antrenare
ax2 = fig.add_subplot(gs[0, 1])
culori_emotii = ['#ff4444','#ff8c00', '#aa44ff', '#ffff44', '#aaaaaa', '#4488ff', '#44ffaa']
bars2 = ax2.bar(emotii_ro, nr_imagini_train, color=culori_emotii,
                edgecolor='#30363d', linewidth=0.8)
ax2.set_title('Distributia Datelor - Emotii (Train)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Emotie', fontsize=10)
ax2.set_ylabel('Numar imagini', fontsize=10)
ax2.set_facecolor('#0d1117')
ax2.tick_params(colors='white', axis='both')
ax2.title.set_color('white')
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
for spine in ax2.spines.values():
    spine.set_edgecolor('#30363d')
for bar, val in zip(bars2, nr_imagini_train):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', va='bottom', fontsize=8, color='white')

#grafic 3: pie chart gesturi
ax3 = fig.add_subplot(gs[1, 0])
ax3.pie(nr_exemple, labels=[g.replace("_"," ") for g in GESTURI],
        colors=culori_gesturi, autopct='%1.1f%%',
        startangle=90, textprops={'color': 'white', 'fontsize': 9})
ax3.set_title('Proportie Clase - Gesturi', fontsize=13, fontweight='bold', pad=10)
ax3.title.set_color('white')
ax3.set_facecolor('#0d1117')

#grafic 4:train vs test emotii
ax4 = fig.add_subplot(gs[1, 1])
x= np.arange(len(emotii_ro))
latime = 0.35
ax4.bar(x -latime/2,nr_imagini_train, latime, label='Train', color='#58a6ff', alpha=0.85)
ax4.bar(x + latime/2,nr_imagini_test, latime, label='Test',color='#3fb950', alpha=0.85)
ax4.set_title('Train vs Test - Emotii', fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('Emotie', fontsize=10)
ax4.set_ylabel('Numar imagini', fontsize=10)
ax4.set_xticks(x)
ax4.set_xticklabels(emotii_ro, rotation=30, ha='right')
ax4.legend(facecolor='#161b22', labelcolor='white')
ax4.set_facecolor('#0d1117')
ax4.tick_params(colors='white')
ax4.title.set_color('white')
ax4.xaxis.label.set_color('white')
ax4.yaxis.label.set_color('white')
for spine in ax4.spines.values():
    spine.set_edgecolor('#30363d')

plt.savefig('distributie_date.png',dpi=300, facecolor='#0d1117', bbox_inches='tight')
plt.show()
print("Salvat: distributie_date.png")