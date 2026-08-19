import time
import tkinter as tk
from collections import deque, Counter

import cv2
from PIL import Image, ImageTk

from src.clasificator_emotii import ClasificatorEmotii
from src.clasificator_gesturi import ClasificatorGesturi
from src.sinteza_vocala import SintezaVocala

#config
PRAG_INCREDERE_GEST = 0.75          #prag ignorare gest
TIMP_CONFIRMARE_GEST = 2.5          #sec. gest -> comanda
LUNGIME_ISTORIC_EMOTII = 7          #cadre vot majoritar

PRAG_ALERTA_DISCONFORT = 7.0       #sec disconfort -> alerta
PRAG_ALERTA_URGENTA = 7.0          #sec pumn inchis -> alerta

GRACE_FATA = 1.5                    #toleranta fata
GRACE_GEST = 1.0                    #toleranta gest

INTERVAL_RE_ALERTA_DISCONFORT = 15  #re-anunt vocal
INTERVAL_RE_ALERTA_URGENTA = 5

CULOARE_FUNDAL = "#0d1117"
CULOARE_CARD = "#161b22"
CULOARE_BORDURA = "#30363d"
CULOARE_TEXT_SECUNDAR = "#8b949e"
CULOARE_TEXT_LABEL = "#484f58"
CULOARE_ACCENT = "#58a6ff"
CULOARE_ALERTA_1 = "#f85149"
CULOARE_ALERTA_2 = "#7a1f1a"


class Aplicatie:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Medical Asistiv")
        #self.root.geometry("1180x760")
        self.root.state("zoomed")
        self.root.configure(bg=CULOARE_FUNDAL)
        #self.root.resizable(False, False)
        self.root.resizable(True, True)

        print("Se incarca modelele...")
        self.detector_emotii = ClasificatorEmotii()
        self.detector_gesturi = ClasificatorGesturi()
        self.tts = SintezaVocala()
        print("Modele incarcate.")

        #emotie/disconfort
        self.istoric_emotii = deque(maxlen=LUNGIME_ISTORIC_EMOTII)
        self.ultima_detectie_fata = time.time()
        self.disconfort_start = None
        self.disconfort_alerta_activa = False
        self.disconfort_ultima_rostire = None

        #gest/comanda
        self.gest_candidat = None
        self.timp_start_candidat = None
        self.gest_deja_rostit = None
        self.istoric_comenzi = deque(maxlen=7)

        #stare urgenta
        self.urgenta_start = None
        self.ultima_detectie_urgenta = time.time()
        self.urgenta_alerta_activa = False
        self.urgenta_ultima_rostire = None

        self.fps_time = time.time()
        self.fps = 0

        #!!!!!!!!!!!!!!
        self.contor_cadre = 0
        self.ultimele_rezultate_emotii = []
        #!!!!!!!!!!!!!!
        #toggle on/off
        self.activ_gest = tk.BooleanVar(value=True)
        self.activ_emotie = tk.BooleanVar(value=True)
        self.activ_alerte = tk.BooleanVar(value=True)
        self.activ_istoric = tk.BooleanVar(value=True)

        # setari reglabile live
        self.var_prag_gest    = tk.DoubleVar(value=0.75)
        self.var_timp_confirmare = tk.DoubleVar(value=2.5)
        self.var_prag_alerta  = tk.DoubleVar(value=7.0)
        #!!!!!!!!!!!!!!

        self._construieste_ui()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.activ = True
        self._update()

    #UI
    def _construieste_ui(self):
        header = tk.Frame(self.root, bg=CULOARE_FUNDAL, height=60)
        header.pack(fill="x", padx=20, pady=(15, 0))

        tk.Label(header, text="SISTEM MEDICAL ASISTIV", font=("Courier", 16, "bold"),
                 bg=CULOARE_FUNDAL, fg=CULOARE_ACCENT).pack(side="left")

        self.lbl_fps = tk.Label(header, text="FPS: 0", font=("Courier", 11),
                                 bg=CULOARE_FUNDAL, fg=CULOARE_TEXT_LABEL)
        self.lbl_fps.pack(side="right", padx=10)

        self.lbl_banner = tk.Label(self.root, text="", font=("Courier", 13, "bold"),
                                    bg=CULOARE_FUNDAL, fg="white", pady=8)
        self.lbl_banner.pack(fill="x", padx=20, pady=(10, 0))

        corp = tk.Frame(self.root, bg=CULOARE_FUNDAL)
        corp.pack(fill="both", expand=True, padx=20, pady=10)

        cadru_video = tk.Frame(corp, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA,
                                highlightthickness=1)
        cadru_video.pack(side="left", padx=(0, 15))
        self.lbl_video = tk.Label(cadru_video, bg=CULOARE_CARD)
        self.lbl_video.pack(padx=2, pady=2)

        # panou = tk.Frame(corp, bg=CULOARE_FUNDAL)
        # panou.pack(side="left", fill="both", expand=True)

        #!!!!!!!!!!!
        panou_container = tk.Frame(corp, bg=CULOARE_FUNDAL)
        panou_container.pack(side="left", fill="both", expand=True)

        canvas_panou = tk.Canvas(panou_container, bg=CULOARE_FUNDAL, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(panou_container, orient="vertical", command=canvas_panou.yview)
        canvas_panou.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas_panou.pack(side="left", fill="both", expand=True)

        panou = tk.Frame(canvas_panou, bg=CULOARE_FUNDAL)
        canvas_panou.create_window((0, 0), window=panou, anchor="nw")

        def _actualizeaza_scroll(event=None):
            canvas_panou.configure(scrollregion=canvas_panou.bbox("all"))
            canvas_panou.itemconfig(1, width=canvas_panou.winfo_width())

        panou.bind("<Configure>", _actualizeaza_scroll)
        canvas_panou.bind("<Configure>", _actualizeaza_scroll)

        def _scroll_mouse(event):
            canvas_panou.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_panou.bind_all("<MouseWheel>", _scroll_mouse)
        #!!!!!!!!!!!!

        self._card_gest(panou)
        tk.Frame(panou, bg="#21262d", height=1).pack(fill="x", pady=10)
        self._card_emotie(panou)
        tk.Frame(panou, bg="#21262d", height=1).pack(fill="x", pady=10)
        self._card_alerte(panou)
        tk.Frame(panou, bg="#21262d", height=1).pack(fill="x", pady=10)
        self._card_istoric(panou)

        #!!!!!!!!!!!!!!!!!!!!!!!!!
        tk.Frame(panou, bg="#21262d", height=1).pack(fill="x", pady=10)
        self._panou_setari(panou)
        #!!!!!!!!!!!!!!!!!!!!!!!!!

        tk.Button(self.root, text="⏹  Oprește aplicația", font=("Courier", 11, "bold"),
                  bg="#21262d", fg="#f85149", activebackground="#2d1b1b",
                  activeforeground="#f85149", bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._inchide).pack(pady=(5, 15))

    #!!!!!!!!!!
    def _adauga_buton_toggle(self, parinte, variabila, culoare_on):
        buton = tk.Button(parinte, font=("Courier", 9, "bold"), bd=0, cursor="hand2", width=5)
        
        def update_vizual():
            if variabila.get():
                buton.config(text="ON", bg=culoare_on, fg="white")
            else:
                buton.config(text="OFF", bg="#30363d", fg="#8b949e") 
                
        def la_click():
            variabila.set(not variabila.get())
            update_vizual()
            
        buton.config(command=la_click)
        update_vizual()
        buton.pack(side="right")
    #!!!!!!!!!!!!!!!

    def _card_gest(self, parinte):  
        cadru = tk.Frame(parinte, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA,
                          highlightthickness=1)
        cadru.pack(fill="x")

        #tk.Label(cadru, text="COMANDA ASISTIVA", font=("Courier", 9, "bold"), bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(anchor="w", padx=15, pady=(10, 2))

        #!!!!!!!!!!!!!!
        antet = tk.Frame(cadru, bg=CULOARE_CARD)
        antet.pack(fill="x", padx=15, pady=(10, 2))
        tk.Label(antet, text="COMANDA ASISTIVA", font=("Courier", 9, "bold"),
                  bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
        # tk.Checkbutton(antet, text="ON/OFF", variable=self.activ_gest, indicatoron=False,
        #                bg="#21262d", fg=CULOARE_ACCENT, selectcolor=CULOARE_BORDURA, 
        #                font=("Courier", 8), cursor="hand2").pack(side="right")
        self._adauga_buton_toggle(antet, self.activ_gest, CULOARE_ACCENT)
        #!!!!!!!!!!!!!!


        self.lbl_gest = tk.Label(cadru, text="—", font=("Courier", 24, "bold"),
                                  bg=CULOARE_CARD, fg=CULOARE_ACCENT)
        self.lbl_gest.pack(anchor="w", padx=15)

        info_frame = tk.Frame(cadru, bg=CULOARE_CARD)
        info_frame.pack(fill="x", padx=15, pady=(4, 4))
        tk.Label(info_frame, text="Confirmare comanda:", font=("Courier", 9),
                  bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
        self.lbl_conf_gest = tk.Label(info_frame, text="0%", font=("Courier", 9, "bold"),
                                        bg=CULOARE_CARD, fg=CULOARE_ACCENT)
        self.lbl_conf_gest.pack(side="right")

        self.bara_gest_bg = tk.Frame(cadru, bg="#21262d", height=6)
        self.bara_gest_bg.pack(fill="x", padx=15, pady=(0, 12))
        self.bara_gest = tk.Frame(self.bara_gest_bg, bg=CULOARE_ACCENT, height=6)
        self.bara_gest.place(x=0, y=0, relheight=1, width=0)

    def _card_emotie(self, parinte):
        cadru = tk.Frame(parinte, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA,
                          highlightthickness=1)
        cadru.pack(fill="x")

        #tk.Label(cadru, text="STARE PACIENT (monitorizare pasiva)", font=("Courier", 9, "bold"), bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(anchor="w", padx=15, pady=(10, 2))

        #!!!!!!!!!!!
        antet = tk.Frame(cadru, bg=CULOARE_CARD)
        antet.pack(fill="x", padx=15, pady=(10, 2))
        tk.Label(antet, text="STARE PACIENT", font=("Courier", 9, "bold"),
                  bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
        # tk.Checkbutton(antet, text="ON/OFF", variable=self.activ_emotie, indicatoron=False,
        #                bg="#21262d", fg="#3fb950", selectcolor=CULOARE_BORDURA, 
        #                font=("Courier", 8), cursor="hand2").pack(side="right")
        self._adauga_buton_toggle(antet, self.activ_emotie, "#3fb950")
        #!!!!!!!!!!!

        self.lbl_emotie = tk.Label(cadru, text="—", font=("Courier", 24, "bold"),
                                    bg=CULOARE_CARD, fg="#3fb950")
        self.lbl_emotie.pack(anchor="w", padx=15)

        self.lbl_emotie_raw = tk.Label(cadru, text="", font=("Courier", 9),
                                        bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR)
        self.lbl_emotie_raw.pack(anchor="w", padx=15, pady=(0, 8))

        pady_bot = 12
        self.bara_emotie_bg = tk.Frame(cadru, bg="#21262d", height=6)
        self.bara_emotie_bg.pack(fill="x", padx=15, pady=(0, pady_bot))
        self.bara_emotie = tk.Frame(self.bara_emotie_bg, bg="#3fb950", height=6)
        self.bara_emotie.place(x=0, y=0, relheight=1, width=0)

    def _card_alerte(self, parinte):
        cadru = tk.Frame(parinte, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA,
                          highlightthickness=1)
        cadru.pack(fill="x")

        #tk.Label(cadru, text="MONITORIZARE ALERTE (sustinute > 7s)", font=("Courier", 9, "bold"), bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(anchor="w", padx=15, pady=(10, 8))
       
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        antet = tk.Frame(cadru, bg=CULOARE_CARD)
        antet.pack(fill="x", padx=15, pady=(10, 8))
        tk.Label(antet, text="MONITORIZARE ALERTE", font=("Courier", 9, "bold"),
                  bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
        # tk.Checkbutton(antet, text="ON/OFF", variable=self.activ_alerte, indicatoron=False,
        #                bg="#21262d", fg=CULOARE_ALERTA_1, selectcolor=CULOARE_BORDURA, 
        #                font=("Courier", 8), cursor="hand2").pack(side="right")
        self._adauga_buton_toggle(antet, self.activ_alerte, CULOARE_ALERTA_1)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        f1 = tk.Frame(cadru, bg=CULOARE_CARD)
        f1.pack(fill="x", padx=15)
        tk.Label(f1, text="Disconfort / Durere sustinut(a)", font=("Courier", 9),
                  bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR).pack(side="left")
        self.lbl_pct_disconfort = tk.Label(f1, text="0%", font=("Courier", 9, "bold"),
                                            bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR)
        self.lbl_pct_disconfort.pack(side="right")

        self.bara_disconfort_bg = tk.Frame(cadru, bg="#21262d", height=6)
        self.bara_disconfort_bg.pack(fill="x", padx=15, pady=(2, 10))
        self.bara_disconfort = tk.Frame(self.bara_disconfort_bg, bg=CULOARE_TEXT_LABEL, height=6)
        self.bara_disconfort.place(x=0, y=0, relheight=1, width=0)

        f2 = tk.Frame(cadru, bg=CULOARE_CARD)
        f2.pack(fill="x", padx=15)
        tk.Label(f2, text="Semnal de urgenta sustinut", font=("Courier", 9),
                  bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR).pack(side="left")
        self.lbl_pct_urgenta = tk.Label(f2, text="0%", font=("Courier", 9, "bold"),
                                         bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR)
        self.lbl_pct_urgenta.pack(side="right")

        self.bara_urgenta_bg = tk.Frame(cadru, bg="#21262d", height=6)
        self.bara_urgenta_bg.pack(fill="x", padx=15, pady=(2, 12))
        self.bara_urgenta = tk.Frame(self.bara_urgenta_bg, bg=CULOARE_TEXT_LABEL, height=6)
        self.bara_urgenta.place(x=0, y=0, relheight=1, width=0)

    def _card_istoric(self, parinte):
        cadru = tk.Frame(parinte, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA,
                          highlightthickness=1)
        cadru.pack(fill="x")

        #tk.Label(cadru, text="ISTORIC COMENZI (ultimele 7)", font=("Courier", 9, "bold"), bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(anchor="w", padx=15, pady=(10, 6))

        #!!!!!!!!!!!!!!!
        antet = tk.Frame(cadru, bg=CULOARE_CARD)
        antet.pack(fill="x", padx=15, pady=(10, 6))
        tk.Label(antet, text="ISTORIC COMENZI", font=("Courier", 9, "bold"),
                  bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
        # tk.Checkbutton(antet, text="ON/OFF", variable=self.activ_istoric, indicatoron=False,
        #                bg="#21262d", fg="white", selectcolor=CULOARE_BORDURA, 
        #                font=("Courier", 8), cursor="hand2").pack(side="right")
        self._adauga_buton_toggle(antet, self.activ_istoric, "#58a6ff")
        #!!!!!!!!!!!!!!!

        self.lbl_istoric = tk.Label(cadru, text="—", font=("Courier", 10),
                                     bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR,
                                     wraplength=340, justify="left")
        self.lbl_istoric.pack(anchor="w", padx=15, pady=(0, 12))

    #!!!!!!!!!!!!!!!!!!!
    def _panou_setari(self, parinte):
        cadru = tk.Frame(parinte, bg=CULOARE_CARD,
                        highlightbackground=CULOARE_BORDURA, highlightthickness=1)
        cadru.pack(fill="x", pady=(0, 0))

        tk.Label(cadru, text="SETARI PACIENT",
                font=("Courier", 9, "bold"),
                bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(anchor="w", padx=15, pady=(10, 6))

        def _slider(parinte_local, eticheta, variabila, val_min, val_max, format_val):
            rand = tk.Frame(parinte_local, bg=CULOARE_CARD)
            rand.pack(fill="x", padx=15, pady=(0, 8))

            lbl_stanga = tk.Label(rand, text=eticheta,
                                font=("Courier", 9), bg=CULOARE_CARD,
                                fg=CULOARE_TEXT_SECUNDAR, width=22, anchor="w")
            lbl_stanga.pack(side="left")

            lbl_val = tk.Label(rand, text=format_val(variabila.get()),
                            font=("Courier", 9, "bold"),
                            bg=CULOARE_CARD, fg=CULOARE_ACCENT, width=6, anchor="e")
            lbl_val.pack(side="right")

            def la_schimbare(_event=None):
                lbl_val.config(text=format_val(variabila.get()))

            tk.Scale(rand, variable=variabila,
                    from_=val_min, to=val_max, resolution=(val_max - val_min) / 100,
                    orient="horizontal", showvalue=False,
                    bg=CULOARE_CARD, fg=CULOARE_ACCENT,
                    highlightthickness=0, troughcolor="#21262d",
                    activebackground=CULOARE_ACCENT,
                    command=la_schimbare).pack(side="left", fill="x", expand=True, padx=(8, 8))

        _slider(cadru, "Incredere minima gest",
                self.var_prag_gest,       0.50, 0.95, lambda v: f"{v:.0%}")
        _slider(cadru, "Timp confirmare (sec)",
                self.var_timp_confirmare, 1.0,  5.0,  lambda v: f"{v:.1f}s")
        _slider(cadru, "Timp alerta (sec)",
                self.var_prag_alerta,     3.0,  15.0, lambda v: f"{v:.0f}s")

        tk.Frame(cadru, bg=CULOARE_BORDURA, height=1).pack(fill="x", padx=15)

        tk.Button(cadru, text="Reseteaza valorile implicite",
                font=("Courier", 8), bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR,
                activebackground="#21262d", activeforeground=CULOARE_ACCENT,
                bd=0, padx=10, pady=6, cursor="hand2",
                command=lambda: [
                    self.var_prag_gest.set(0.75),
                    self.var_timp_confirmare.set(2.5),
                    self.var_prag_alerta.set(7.0),
                ]).pack(anchor="e", padx=15, pady=8)
    #!!!!!!!!!!!!!!!!!!!

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def _anuleaza_ultima_comanda(self):
        self.urgenta_alerta_activa = False
        self.urgenta_start = None
        self.urgenta_ultima_rostire = None
            
        self.disconfort_alerta_activa = False
        self.disconfort_start = None
        self.disconfort_ultima_rostire = None

        if len(self.istoric_comenzi) > 0:
            self.istoric_comenzi.pop()
            print("[!] Ultima comanda a fost stearsa, iar alertele au fost oprite.")
        else:
            print("[!] Istoric gol. Nu exista comenzi de anulat.")
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #LOOP PRINCIPAL
    def _update(self):
        if not self.activ:
            return

        succes, frame = self.cap.read()
        if not succes:
            self.root.after(30, self._update)
            return

        frame = cv2.flip(frame, 1)
        now = time.time()

        self.fps = 1.0 / (now - self.fps_time + 1e-9)
        self.fps_time = now

        stare_afisata = {"comanda": "—", "progres_gest": 0.0, "incredere_gest": 0.0,
                          "emotie_ro": "—", "emotie_hex": "#3fb950", "incredere_emotie": 0.0}

        #landmarks, gest_raw, comanda_en, incredere_gest = self.detector_gesturi.proceseaza_cadru(frame)
        #!!!!!!!!!!!!!!!!
        if self.activ_gest.get():
            landmarks, gest_raw, comanda_en, incredere_gest = self.detector_gesturi.proceseaza_cadru(frame)
        else:
            landmarks, gest_raw, comanda_en, incredere_gest = None, None, None, 0.0
            self.gest_candidat = None
            self.timp_start_candidat = None
        #!!!!!!!!!!!!!!!

        #gest_valid = landmarks is not None and incredere_gest > PRAG_INCREDERE_GEST
        #!!!!!!!!!!!!!!
        gest_valid = landmarks is not None and incredere_gest > self.var_prag_gest.get()
        #!!!!!!!!!!!!!!

        if gest_valid:
            self.detector_gesturi.deseneaza_palma(frame, landmarks)

            if comanda_en != self.gest_candidat:
                self.gest_candidat = comanda_en
                self.timp_start_candidat = now

            timp_tinut = now - self.timp_start_candidat
            #progres = min(timp_tinut / TIMP_CONFIRMARE_GEST, 1.0)
            #!!!!!!!!!!!!!
            progres = min(timp_tinut / self.var_timp_confirmare.get(), 1.0)
            #!!!!!!!!!!!!!

            if progres >= 1.0 and self.gest_deja_rostit != comanda_en:
                self.tts.roteste(comanda_en)
                self.gest_deja_rostit = comanda_en
                #self.istoric_comenzi.append(comanda_en)
                #!!!!!!!!!
                if comanda_en == 'Cancel command.':
                    self._anuleaza_ultima_comanda()
                else:
                    if self.activ_istoric.get():
                        self.istoric_comenzi.append(comanda_en)
                #!!!!!!!!!

            stare_afisata["comanda"] = comanda_en
            stare_afisata["progres_gest"] = progres
            stare_afisata["incredere_gest"] = incredere_gest

            #cv2.putText(frame, comanda_en, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 166, 255), 2, cv2.LINE_AA)
        else:
            self.gest_candidat = None
            self.timp_start_candidat = None
            self.gest_deja_rostit = None

        #!!!!!!!!!!!!!!!!!!!
        if not self.activ_alerte.get():
            self.urgenta_start = None
            self.urgenta_alerta_activa = False
            self.disconfort_start = None
            self.disconfort_alerta_activa = False

        gest_e_urgenta = gest_valid and gest_raw == ClasificatorGesturi.GEST_URGENTA and self.activ_alerte.get()
        #!!!!!!!!!!!!!

        #gest_e_urgenta = gest_valid and gest_raw == ClasificatorGesturi.GEST_URGENTA
        # if gest_e_urgenta:
        #     if self.urgenta_start is None:
        #         self.urgenta_start = now
        #     self.ultima_detectie_urgenta = now
        # elif self.urgenta_start is not None and (now - self.ultima_detectie_urgenta) > GRACE_GEST:
        #     self.urgenta_start = None
        #     self.urgenta_alerta_activa = False
        #     self.urgenta_ultima_rostire = None
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if gest_e_urgenta:
            if self.urgenta_start is None:
                self.urgenta_start = now
            self.ultima_detectie_urgenta = now
        elif self.urgenta_start is not None and (now - self.ultima_detectie_urgenta) > GRACE_GEST:
            if not self.urgenta_alerta_activa:
                self.urgenta_start = None
                self.urgenta_ultima_rostire = None
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        progres_urgenta = 0.0
        if self.urgenta_start is not None:
            #progres_urgenta = min((now - self.urgenta_start) / PRAG_ALERTA_URGENTA, 1.0)
            #!!!!!!!!!!!!!
            progres_urgenta = min((now - self.urgenta_start) / self.var_prag_alerta.get(), 1.0)
            #!!!!!!!!!!!!!
            if progres_urgenta >= 1.0:
                self.urgenta_alerta_activa = True
                if (self.urgenta_ultima_rostire is None or
                        now - self.urgenta_ultima_rostire > INTERVAL_RE_ALERTA_URGENTA):
                    self.tts.roteste_prioritar(
                        "Critical alert. Emergency signal sustained. Please respond immediately."
                    )
                    self.urgenta_ultima_rostire = now

        #v1
        # rezultate_emotii = self.detector_emotii.proceseaza_cadru(frame)

        # if rezultate_emotii:
        #     self.ultima_detectie_fata = now

        #v2
        # self.contor_cadre += 1
        
        # if self.contor_cadre % 3 == 0:
        #     self.ultimele_rezultate_emotii = self.detector_emotii.proceseaza_cadru(frame)
            
        # rezultate_emotii = self.ultimele_rezultate_emotii

        # if rezultate_emotii:
        #     self.ultima_detectie_fata = now

        #v3
        #!!!!!!!!!!!!!!!!!!
        if self.activ_emotie.get():
            self.contor_cadre += 1
            if self.contor_cadre % 3 == 0:
                self.ultimele_rezultate_emotii = self.detector_emotii.proceseaza_cadru(frame)
            rezultate_emotii = self.ultimele_rezultate_emotii
        else:
            rezultate_emotii = []

        if rezultate_emotii:
            self.ultima_detectie_fata = now
        #!!!!!!!!!!!!!!!!!

        for res in rezultate_emotii:
            x, y, w, h = res['box']
            self.istoric_emotii.append((res['stare'], res['culoare_bgr'], res['culoare_hex']))

            stare_stabila, culoare_bgr, culoare_hex = Counter(self.istoric_emotii).most_common(1)[0][0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), culoare_bgr, 2, cv2.LINE_AA)
            text_afisat = f"{stare_stabila} ({res['incredere']*100:.0f}%)"
            cv2.rectangle(frame, (x, y - 30), (x + w, y), culoare_bgr, -1)
            cv2.putText(frame, text_afisat, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 0, 0) if culoare_bgr != (0, 0, 255) else (255, 255, 255), 2, cv2.LINE_AA)

            #stare_afisata["emotie_ro"] = f"{res['emotie_ro'].upper()} {stare_stabila}"
            stare_afisata["emotie_ro"] = f"{stare_stabila}({res['emotie_ro']})"
            stare_afisata["emotie_hex"] = culoare_hex
            stare_afisata["incredere_emotie"] = res['incredere']

        fata_prezenta = (now - self.ultima_detectie_fata) <= GRACE_FATA
        stare_curenta = None
        if rezultate_emotii:
            stare_curenta = Counter(self.istoric_emotii).most_common(1)[0][0][0]

        if fata_prezenta and stare_curenta == 'Discomfort / Pain':
            if fata_prezenta and stare_curenta == 'Discomfort/Pain' and self.activ_alerte.get():
                if self.disconfort_start is None:
                    self.disconfort_start = now
            else:
                self.disconfort_start = None
                self.disconfort_alerta_activa = False
                self.disconfort_ultima_rostire = None
        
        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if fata_prezenta and stare_curenta == 'Comfort':
        #     self.disconfort_start = None
        #     self.disconfort_alerta_activa = False
        #     self.disconfort_ultima_rostire = None
        # elif fata_prezenta and stare_curenta in ['Discomfort/Pain', 'Discomfort'] and self.activ_alerte.get():
        #     if self.disconfort_start is None:
        #         self.disconfort_start = now
        # else:
        #     if not self.disconfort_alerta_activa:
        #         self.disconfort_start = None
        #         self.disconfort_ultima_rostire = None
        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        progres_disconfort = 0.0
        if self.disconfort_start is not None:
            #progres_disconfort = min((now - self.disconfort_start) / PRAG_ALERTA_DISCONFORT, 1.0)
            #!!!!!!!!!!!!!
            progres_disconfort = min((now - self.disconfort_start) / self.var_prag_alerta.get(), 1.0)
            #!!!!!!!!!!!!!
            if progres_disconfort >= 1.0:
                self.disconfort_alerta_activa = True
                if (self.disconfort_ultima_rostire is None or
                        now - self.disconfort_ultima_rostire > INTERVAL_RE_ALERTA_DISCONFORT):
                    self.tts.roteste_prioritar(
                        "Attention. Patient has been showing signs of discomfort for an extended period."
                    )
                    self.disconfort_ultima_rostire = now

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_redim = cv2.resize(frame_rgb, (640, 480))
        img = ImageTk.PhotoImage(Image.fromarray(frame_redim))
        self.lbl_video.imgtk = img
        self.lbl_video.configure(image=img)

        self._update_ui(stare_afisata, progres_disconfort, progres_urgenta)

        self.root.after(10, self._update)

    #PANOURI LATERALE
    def _update_ui(self, stare, progres_disconfort, progres_urgenta):
        
        self.lbl_gest.config(text=stare["comanda"])
        self.lbl_conf_gest.config(text=f"{stare['progres_gest']*100:.0f}%")
        latime = self.bara_gest_bg.winfo_width()
        culoare_bara_gest = "#3fb950" if stare['progres_gest'] >= 1.0 else CULOARE_ACCENT
        self.bara_gest.config(bg=culoare_bara_gest)
        self.bara_gest.place(width=int(stare['progres_gest'] * latime))

        self.lbl_emotie.config(text=stare["emotie_ro"], fg=stare["emotie_hex"])
        self.lbl_emotie_raw.config(text=f"Incredere model: {stare['incredere_emotie']*100:.0f}%")
        latime_e = self.bara_emotie_bg.winfo_width()
        self.bara_emotie.config(bg=stare["emotie_hex"])
        self.bara_emotie.place(width=int(stare['incredere_emotie'] * latime_e))

        self._seteaza_bara_alerta(self.bara_disconfort, self.bara_disconfort_bg,
                                   self.lbl_pct_disconfort, progres_disconfort, self.disconfort_alerta_activa)
        self._seteaza_bara_alerta(self.bara_urgenta, self.bara_urgenta_bg,
                                   self.lbl_pct_urgenta, progres_urgenta, self.urgenta_alerta_activa)

        flash = int(time.time() * 2) % 2 == 0
        if self.urgenta_alerta_activa:
            self.lbl_banner.config(
                text="CRITICAL: semnal de urgenta sustinut",
                bg=CULOARE_ALERTA_1 if flash else CULOARE_ALERTA_2, fg="white")
        elif self.disconfort_alerta_activa:
            self.lbl_banner.config(
                text="ALERTA: disconfort sustinut de peste 7 secunde",
                bg=CULOARE_ALERTA_1 if flash else CULOARE_ALERTA_2, fg="white")
        else:
            self.lbl_banner.config(text="", bg=CULOARE_FUNDAL)

        #istoric comenzi
        if self.istoric_comenzi:
            self.lbl_istoric.config(text=" → ".join(self.istoric_comenzi))
        else:
            self.lbl_istoric.config(text="—")

        self.lbl_fps.config(text=f"FPS: {self.fps:.0f}")

    @staticmethod
    def _seteaza_bara_alerta(bara, bara_bg, label_pct, progres, activa):
        latime = bara_bg.winfo_width()
        culoare = CULOARE_ALERTA_1 if activa else ("#d29922" if progres > 0 else CULOARE_TEXT_LABEL)
        bara.config(bg=culoare)
        bara.place(width=int(progres * latime))
        label_pct.config(text=f"{progres*100:.0f}%", fg=culoare)

    def _inchide(self):
        self.activ = False
        self.cap.release()
        self.root.destroy()


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = Aplicatie(root)
#     root.mainloop()

if __name__ == "__main__":
    from src.ecran_pornire import EcranPornire

    root = tk.Tk()
    root.withdraw() 

    def porneste_aplicatia():
        root.deiconify()
        Aplicatie(root)

    EcranPornire(root, porneste_aplicatia)
    root.mainloop()