import os
import time
import cv2
from datetime import datetime


class CapturadorImagenes:
    def __init__(self):
        self.contador = 0

    def capturar(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo acceder a la cámara")

        ret, frame = cap.read()
        cap.release()

        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            nombre_archivo = f"capturas/captura_{timestamp}.png"
            cv2.imwrite(nombre_archivo, frame)
            self.contador += 1
            print(f"\r[+] Capturas realizadas: {self.contador}", end='')
