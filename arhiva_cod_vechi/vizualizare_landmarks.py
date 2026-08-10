import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

GESTURI =["pumn_inchis","palma_deschisa", "unu", "doi", "trei", "ok"]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

fig, axs =plt.subplots(2, 3, figsize=(14, 9))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Vizualizare Landmarks Mediapipe — Exemple per Gest',
             fontsize=15, fontweight='bold', color='white', y=1.01)

culori_degete = {
    range(0,1): '#ffffff',   #incheietura
    range(1,5): '#58a6ff',   #deget mare
    range(5,9): '#3fb950',   #aratator
    range(9,13): '#ff8c00',  #mijlociu
    range(13,17): '#aa44ff', #inelar
    range(17,21): '#ff4444', #mic
}

def culoare_punct(idx):
    for r, c in culori_degete.items():
        if idx in r:
            return c
    return 'white'

for ax, gest in zip(axs.flat, GESTURI):
    ax.set_facecolor('#161b22')
    ax.set_title(gest.replace("_", " ").upper(),
                 fontsize=10, fontweight='bold', color='#58a6ff', pad=6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    df = pd.read_csv(f"date_gesturi/{gest}.csv", header=None)

    #desenam primele 5 exemple suprapuse (transparenta)
    for _, rand in df.head(5).iterrows():
        vals= rand.values
        puncte = vals.reshape(21, 3)
        xs =puncte[:, 0]
        ys = puncte[:, 1]

        for start,end in HAND_CONNECTIONS:
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]],
                   color='#30363d', linewidth=1.5, alpha=0.6, zorder=1)

        for i in range(21):
            c = culoare_punct(i)
            ax.scatter(xs[i], ys[i], color=c, s=30, zorder=2, alpha=0.8)

    #desenam media exemplelor (linia principala)
    medie= df.mean().values.reshape(21, 3)
    xs_m, ys_m =medie[:, 0], medie[:, 1]

    for start, end in HAND_CONNECTIONS:
        ax.plot([xs_m[start], xs_m[end]], [ys_m[start], ys_m[end]],
               color='white', linewidth=2, zorder=3)
    for i in range(21):
        c = culoare_punct(i)
        ax.scatter(xs_m[i], ys_m[i], color=c, s=60, zorder=4, edgecolors='white', linewidth=0.5)

#legenda degete
legenda_items = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#58a6ff', markersize=8, label='Deget mare'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3fb950', markersize=8, label='Aratator'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff8c00', markersize=8, label='Mijlociu'),
    plt.Line2D([0],[0], marker='o',color='w', markerfacecolor='#aa44ff', markersize=8, label='Inelar'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff4444', markersize=8, label='Degetul mic'),
]
fig.legend(handles=legenda_items,loc='lower center', ncol=5,
           facecolor='#161b22', labelcolor='white', fontsize=9,
           framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('vizualizare_landmarks.png', dpi=300, facecolor='#0d1117', bbox_inches='tight')
plt.show()
print("Salvat: vizualizare_landmarks.png")