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
        self.hilo_captura = None

    def procesar_dataset(self):
        self._mostrar_mensaje("Procesando dataset completo...")
        self.procesador.procesar_directorio('./datasets', './salidas/dataset')
        self._mostrar_mensaje("✅ Dataset procesado guardado en 'salidas/dataset'")

    def procesar_capturas(self):
        self._mostrar_mensaje("Iniciando captura en tiempo real...")
        self.ejecutando = True

        def capturar():
            while self.ejecutando:
                self.capturador.capturar()
                time.sleep(0.5)
            self.procesador.procesar_directorio('capturas', './salidas/camara')
            self._mostrar_mensaje("✅ Capturas procesadas guardadas en 'salidas/camara'")

        self.hilo_captura = threading.Thread(target=capturar)
        self.hilo_captura.start()

    def detener_captura(self):
        if self.ejecutando:
            self.ejecutando = False
            if self.hilo_captura:
                self.hilo_captura.join()
            self._mostrar_mensaje("⏹️ Captura detenida.")

    def extraer_features_dataset(self):
        self._mostrar_mensaje("Extrayendo características del dataset...")
        self.extractor.extraer_de_directorio(
            'datasets', 'features/dataset_features.csv')
        self._mostrar_mensaje("✅ Características guardadas en 'features/dataset_features.csv'")

    def extraer_features_capturas(self):
        self._mostrar_mensaje("Extrayendo características de capturas...")
        self.extractor.extraer_de_directorio(
            './salidas/camara', './features/capturas_features.csv')
        self._mostrar_mensaje("✅ Características guardadas en 'features/capturas_features.csv'")

    def _mostrar_mensaje(self, mensaje):
        print(mensaje)


def lanzar_consola():
    sistema = SistemaVision()

    while True:
        print("\n=== Menú Principal ===")
        print("1. 📁 Procesar dataset completo")
        print("2. 📷 Capturar imágenes en tiempo real")
        print("3. ⏹️ Detener captura")
        print("4. 🧬 Extraer características del dataset")
        print("5. 🧪 Extraer características de capturas")
        print("6. ❌ Salir")

        opcion = input("Selecciona una opción (1-6): ")

        if opcion == '1':
            sistema.procesar_dataset()
        elif opcion == '2':
            sistema.procesar_capturas()
        elif opcion == '3':
            sistema.detener_captura()
        elif opcion == '4':
            sistema.extraer_features_dataset()
        elif opcion == '5':
            sistema.extraer_features_capturas()
        elif opcion == '6':
            sistema.detener_captura()
            print("Saliendo del sistema.")
            break
        else:
            print("❌ Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    lanzar_consola()
