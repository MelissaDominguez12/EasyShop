import os
import cv2
import numpy as np


class ProcesadorImagenes:
    def __init__(self):
        self.registro_procesadas = set()

    def _cargar_registro_procesadas(self, salida_dir):
        registro_path = os.path.join(salida_dir, 'registro_procesadas.txt')
        if os.path.exists(registro_path):
            with open(registro_path, 'r') as f:
                self.registro_procesadas = set(f.read().splitlines())

    def _guardar_registro_procesadas(self, salida_dir):
        registro_path = os.path.join(salida_dir, 'registro_procesadas.txt')
        with open(registro_path, 'w') as f:
            for nombre in self.registro_procesadas:
                f.write(f"{nombre}\n")

    def _segmentar_objeto(self, img_path):
        """Segmenta el objeto principal con fondo blanco o casi blanco garantizado"""
        try:
            # Leer imagen
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("No se pudo leer la imagen")

            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Umbralización para detectar el fondo blanco (fondo claro)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

            # Operaciones morfológicas para limpiar la máscara
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Encontrar los contornos
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Crear máscara del objeto principal
            mask = np.zeros_like(gray)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -
                             1, 255, thickness=cv2.FILLED)

            # Aplicar máscara a la imagen original
            result = cv2.bitwise_and(img, img, mask=mask)

            # Convertir áreas fuera de la máscara a negro absoluto
            result[mask == 0] = [0, 0, 0]

            return result

        except Exception as e:
            print(f"\n[!] Error en segmentación: {str(e)}")
            return None

    def procesar_imagen(self, entrada_path, salida_path, clase):
        try:
            nombre_archivo = os.path.basename(entrada_path)
            identificador = f"{clase}_{nombre_archivo}"

            if identificador in self.registro_procesadas:
                return False

            # Segmentar objeto
            segmented = self._segmentar_objeto(entrada_path)
            if segmented is None:
                print(f"\n[!] No se pudo segmentar {entrada_path}")
                return False

            # Guardar resultado
            cv2.imwrite(salida_path, segmented)
            self.registro_procesadas.add(identificador)
            return True

        except Exception as e:
            print(f"\n[!] Error procesando {entrada_path}: {str(e)}")
            return False

    def procesar_directorio(self, entrada_dir, salida_dir):
        self._cargar_registro_procesadas(salida_dir)

        archivos_procesados = 0
        total_archivos = 0
        errores = 0

        # Contar archivos válidos
        for clase in os.listdir(entrada_dir):
            clase_dir = os.path.join(entrada_dir, clase)
            if os.path.isdir(clase_dir):
                for archivo in os.listdir(clase_dir):
                    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                        total_archivos += 1

        # Procesar imágenes
        for clase in os.listdir(entrada_dir):
            clase_dir = os.path.join(entrada_dir, clase)
            if not os.path.isdir(clase_dir):
                continue

            clase_salida = os.path.join(salida_dir, clase)
            os.makedirs(clase_salida, exist_ok=True)

            for archivo in os.listdir(clase_dir):
                if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                    entrada_path = os.path.join(clase_dir, archivo)
                    nombre_base = os.path.splitext(archivo)[0]
                    salida_path = os.path.join(
                        clase_salida, f"seg_{nombre_base}.png")

                    if self.procesar_imagen(entrada_path, salida_path, clase):
                        archivos_procesados += 1
                    else:
                        errores += 1

                    print(
                        f"\r[+] Progreso: {archivos_procesados}/{total_archivos} | Errores: {errores}", end='')

        self._guardar_registro_procesadas(salida_dir)
        print(
            f"\n[+] Proceso finalizado. Éxitos: {archivos_procesados}, Fallos: {errores}")
