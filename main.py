import os
import time
import threading
from utils import crear_directorios
from camara import CapturadorImagenes
from procesamiento import ProcesadorImagenes
from caracteristicas import ExtractorCaracteristicas


class SistemaVision:
    def __init__(self):
        crear_directorios()
        self.capturador = CapturadorImagenes()
        self.procesador = ProcesadorImagenes()
        self.extractor = ExtractorCaracteristicas()
        self.ejecutando = False

    def mostrar_menu(self):
        print("\n=== SISTEMA DE VISIÓN POR COMPUTADORA ===")
        print("1. Procesar dataset completo")
        print("2. Capturar y procesar imágenes en tiempo real")
        print("3. Extraer características de dataset")
        print("4. Extraer características de capturas")
        print("5. Salir")

    def procesar_dataset(self):
        print("\n[+] Procesando dataset...")
        self.procesador.procesar_directorio('./datasets', './salidas/dataset')
        print("[+] Dataset procesado guardado en 'salidas/dataset'")

    def procesar_capturas(self):
        print("\n[+] Iniciando captura en tiempo real (Presione 'q' para detener)")
        self.ejecutando = True

        def capturar():
            while self.ejecutando:
                self.capturador.capturar()
                time.sleep(0.5)  # 2 FPS

        hilo_captura = threading.Thread(target=capturar)
        hilo_captura.start()

        input()  # Esperar entrada para detener
        self.ejecutando = False
        hilo_captura.join()

        print("\n[+] Procesando capturas...")
        self.procesador.procesar_directorio('capturas', './salidas/camara')
        print("[+] Capturas procesadas guardadas en 'salidas/camara'")

    def extraer_features_dataset(self):
        print("\n[+] Extrayendo características del dataset...")
        # Ahora pasamos la ruta raíz del dataset (que contiene las carpetas por clase)
        self.extractor.extraer_de_directorio(
            'datasets', 'features/dataset_features.csv')
        print("[+] Características guardadas en 'features/dataset_features.csv'")

    def extraer_features_capturas(self):
        print("\n[+] Extrayendo características de capturas...")
        self.extractor.extraer_de_directorio(
            './salidas/camara', './features/capturas_features.csv')
        print("[+] Características guardadas en 'features/capturas_features.csv'")

    def ejecutar(self):
        while True:
            self.mostrar_menu()
            opcion = input("Seleccione una opción: ")

            if opcion == '1':
                self.procesar_dataset()
            elif opcion == '2':
                self.procesar_capturas()
            elif opcion == '3':
                self.extraer_features_dataset()
            elif opcion == '4':
                self.extraer_features_capturas()
            elif opcion == '5':
                print("\n[+] Saliendo del sistema...")
                break
            else:
                print("\n[!] Opción no válida")


if __name__ == "__main__":
    sistema = SistemaVision()
    sistema.ejecutar()
