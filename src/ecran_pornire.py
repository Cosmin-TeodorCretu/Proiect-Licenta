import tkinter as tk


GESTURI_GHID = [
    ("Pumn inchis", "MEDICAL EMERGENCY!", "#f85149"),
    ("Ok", "I need water.", "#58a6ff"),
    ("Unu", "I am in pain.\nNeed painkillers.", "#f0883e"),
    ("Doi", "Please call a nurse.", "#bc8cff"),
    ("Trei","Adjust the bed, please.", "#3fb950"),
    ("Palma deschisa", "Cancel command.", "#8b949e"),
]

CULOARE_FUNDAL = "#0d1117"
CULOARE_CARD = "#161b22"
CULOARE_BORDURA = "#30363d"
CULOARE_ACCENT = "#58a6ff"
CULOARE_GRI = "#8b949e"
CULOARE_URGENTA = "#f85149"


class EcranPornire(tk.Toplevel):

    def __init__(self, root, callback_pornire):
        super().__init__(root)
        self.root_ref = root
        self.callback_pornire = callback_pornire

        self.title("Sistem Medical Asistiv - Ghid de utilizare")
        self.geometry("820x640")
        self.configure(bg=CULOARE_FUNDAL)
        self.resizable(False, False)

        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 820) // 2
        y = (self.winfo_screenheight() - 640) // 2
        self.geometry(f"820x640+{x}+{y}")

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._inchide_tot)

        self._construieste_ui()

    def _construieste_ui(self):
        tk.Frame(self, bg=CULOARE_ACCENT, height=4).pack(fill="x")

        header = tk.Frame(self, bg=CULOARE_FUNDAL)
        header.pack(fill="x", padx=30, pady=(20, 0))

        tk.Label(
            header,
            text="SISTEM MEDICAL ASISTIV",
            font=("Courier", 18, "bold"),
            bg=CULOARE_FUNDAL, fg=CULOARE_ACCENT,
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Comunicare non-verbala pentru pacienti cu mobilitate redusa",
            font=("Courier", 10),
            bg=CULOARE_FUNDAL, fg=CULOARE_GRI,
        ).pack(anchor="w", pady=(4, 0))

        tk.Frame(self, bg=CULOARE_BORDURA, height=1).pack(fill="x", padx=30, pady=16)

        tk.Label(
            self,
            text="GHID GESTURI - pozitionati mana in fata camerei si mentineti gestul 2-3 secunde",
            font=("Courier", 9, "bold"),
            bg=CULOARE_FUNDAL, fg=CULOARE_GRI,
        ).pack(anchor="w", padx=30, pady=(0, 10))

        grid = tk.Frame(self, bg=CULOARE_FUNDAL)
        grid.pack(padx=30, fill="x")

        for col in range(3):
            grid.columnconfigure(col, weight=1, uniform="col")

        for i, (nume, comanda, culoare) in enumerate(GESTURI_GHID):
            rand = i // 3
            col  = i % 3
            self._card_gest(grid, nume, comanda, culoare).grid(
                row=rand, column=col, padx=6, pady=6, sticky="nsew"
            )

        tk.Frame(self, bg=CULOARE_BORDURA, height=1).pack(fill="x", padx=30, pady=(18, 12))

        nota = tk.Frame(self, bg=CULOARE_CARD,
                        highlightbackground=CULOARE_URGENTA, highlightthickness=1)
        nota.pack(fill="x", padx=30)

        tk.Label(
            nota,
            text="URGENTA: tineti pumnul inchis continuu timp de 7 secunde "
                 "pentru a activa alarma de urgenta cu sinteza vocala automata.",
            font=("Courier", 9),
            bg=CULOARE_CARD, fg=CULOARE_URGENTA,
            wraplength=740, justify="left", pady=10, padx=12,
        ).pack(anchor="w")

        tk.Button(
            self,
            text="Porneste sistemul",
            font=("Courier", 12, "bold"),
            bg=CULOARE_ACCENT, fg=CULOARE_FUNDAL,
            activebackground="#79c0ff", activeforeground=CULOARE_FUNDAL,
            bd=0, padx=24, pady=10, cursor="hand2",
            command=self._porneste,
        ).pack(pady=20)

    def _card_gest(self, parinte, nume, comanda, culoare):
        card = tk.Frame(
            parinte, bg=CULOARE_CARD,
            highlightbackground=culoare, highlightthickness=1,
        )

        tk.Label(
            card, text=nume,
            font=("Courier", 13, "bold"),
            bg=CULOARE_CARD, fg=culoare,
        ).pack(pady=(18, 4))

        tk.Frame(card, bg=culoare, height=1).pack(fill="x", padx=20, pady=6)

        tk.Label(
            card, text=comanda,
            font=("Courier", 9),
            bg=CULOARE_CARD, fg=CULOARE_GRI,
            wraplength=200, justify="center",
        ).pack(pady=(0, 18), padx=10)

        return card

    def _porneste(self):
        self.destroy()
        self.callback_pornire()

    def _inchide_tot(self):
        self.root_ref.destroy()