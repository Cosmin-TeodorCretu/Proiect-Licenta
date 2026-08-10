import pyttsx3
import threading
import queue

class SintezaVocala:
    def __init__(self, viteza=150):
        self.coada_mesaje = queue.Queue()
        self.viteza = viteza

        self.worker = threading.Thread(target=self._proceseaza_voce, daemon=True)
        self.worker.start()

    def _proceseaza_voce(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        while True:
            text = self.coada_mesaje.get()
            if text is None:
                break

            engine = pyttsx3.init()
            engine.setProperty('rate', self.viteza)

            vocale = engine.getProperty('voices')
            for voce in vocale:
                if "english" in voce.name.lower() or "en" in voce.id.lower():
                    engine.setProperty('voice', voce.id)
                    break

            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine

            self.coada_mesaje.task_done()

    def roteste(self, text_engleza):
        self.coada_mesaje.put(text_engleza)

    def roteste_prioritar(self, text_engleza):
        with self.coada_mesaje.mutex:
            self.coada_mesaje.queue.clear()
        self.coada_mesaje.put(text_engleza)

def opreste(self):
        self.coada_mesaje.put(None)