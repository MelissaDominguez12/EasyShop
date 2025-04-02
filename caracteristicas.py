import os
import csv
import numpy as np
from PIL import Image
from utils import imagen_a_array, aplicar_kernel


class ExtractorCaracteristicas:
    def __init__(self):
        self.cabeceras = [
            'clase', 'nombre_archivo', 'ancho', 'alto', 'relacion_aspecto',
            'intensidad_promedio', 'desviacion_intensidad',
            'entropia', 'uniformidad', 'contraste',
            'hu_momento_1', 'hu_momento_2', 'hu_momento_3',
            'hu_momento_4', 'hu_momento_5', 'hu_momento_6', 'hu_momento_7',
            'area_objetos', 'perimetro_promedio', 'compacidad_promedio',
            'excentricidad_promedio', 'solidez_promedio'
        ]
        self.registro_extraidas = set()
        self.kernel_gaussiano = np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]]) / 16
        self.kernel_sobel_x = np.array([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]])
        self.kernel_sobel_y = np.array([[-1, -2, -1],
                                        [0, 0, 0],
                                        [1, 2, 1]])

    def _cargar_registro_extraidas(self, archivo_csv):
        if os.path.exists(archivo_csv):
            with open(archivo_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Saltar cabeceras
                for row in reader:
                    if row and len(row) > 1:
                        self.registro_extraidas.add(f"{row[0]}_{row[1]}")

    def _calcular_momentos_hu(self, imagen):
        """Calcula los 7 momentos de Hu manualmente"""
        def momento_central(p, q, img):
            m00 = np.sum(img)
            if m00 == 0:
                return 0.0

            y_indices, x_indices = np.indices(img.shape)
            centro_x = np.sum(x_indices * img) / m00
            centro_y = np.sum(y_indices * img) / m00

            total = 0.0
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    total += ((x - centro_x) ** p) * \
                        ((y - centro_y) ** q) * img[y, x]
            return total / (m00 ** ((p + q) / 2 + 1))

        n20 = momento_central(2, 0, imagen)
        n02 = momento_central(0, 2, imagen)
        n11 = momento_central(1, 1, imagen)
        n30 = momento_central(3, 0, imagen)
        n12 = momento_central(1, 2, imagen)
        n21 = momento_central(2, 1, imagen)
        n03 = momento_central(0, 3, imagen)

        h1 = n20 + n02
        h2 = (n20 - n02)**2 + 4*n11**2
        h3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
        h4 = (n30 + n12)**2 + (n21 + n03)**2
        h5 = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) + \
             (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
        h6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + \
            4*n11*(n30 + n12)*(n21 + n03)
        h7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) - \
             (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)

        return [h1, h2, h3, h4, h5, h6, h7]

    def _preprocesar_imagen(self, imagen):
        """Preprocesamiento manual de la imagen"""
        array_img = imagen_a_array(imagen)

        # Paso 1: Filtrado Gaussiano
        suavizada = aplicar_kernel(array_img, self.kernel_gaussiano)

        # Paso 2: Ecualización de histograma manual
        hist, _ = np.histogram(suavizada.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        ecualizada = cdf_normalized[suavizada.astype(np.uint8)]

        # Paso 3: Detección de bordes con Sobel
        grad_x = aplicar_kernel(ecualizada, self.kernel_sobel_x)
        grad_y = aplicar_kernel(ecualizada, self.kernel_sobel_y)
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        magnitud = (magnitud / magnitud.max() * 255).astype(np.uint8)

        return magnitud

    def extraer_de_imagen(self, imagen_path, clase):
        try:
            nombre_archivo = os.path.basename(imagen_path)
            identificador = f"{clase}_{nombre_archivo}"

            if identificador in self.registro_extraidas:
                return None

            with Image.open(imagen_path) as img:
                # Características básicas
                ancho, alto = img.size
                relacion_aspecto = ancho / alto

                # Preprocesamiento
                img_procesada = self._preprocesar_imagen(img)

                # Características de intensidad
                intensidad_promedio = np.mean(img_procesada)
                desviacion_intensidad = np.std(img_procesada)

                # Características de textura
                hist, _ = np.histogram(img_procesada.flatten(), 256, [0, 256])
                hist = hist / hist.sum()
                entropia = -np.sum(hist * np.log2(hist + 1e-10))
                uniformidad = np.sum(hist**2)

                # Contraste
                dif_h = np.abs(img_procesada[:, 1:] - img_procesada[:, :-1])
                dif_v = np.abs(img_procesada[1:, :] - img_procesada[:-1, :])
                contraste = (np.mean(dif_h) + np.mean(dif_v)) / 2

                # Binarización para momentos de Hu
                umbral = np.mean(img_procesada)
                img_bin = np.where(img_procesada > umbral, 1, 0)
                momentos_hu = self._calcular_momentos_hu(img_bin)

                # Características de objetos
                area_objetos = np.sum(img_bin)
                perimetro = np.sum(img_bin[:, 1:] != img_bin[:, :-1]) + \
                    np.sum(img_bin[1:, :] != img_bin[:-1, :])
                compacidad = (perimetro ** 2) / (4 * np.pi *
                                                 area_objetos) if area_objetos > 0 else 0

                caracteristicas = [
                    clase, nombre_archivo, ancho, alto, relacion_aspecto,
                    intensidad_promedio, desviacion_intensidad,
                    entropia, uniformidad, contraste,
                    *momentos_hu,
                    area_objetos, perimetro, compacidad,
                    0, 0  # Placeholder para excentricidad y solidez
                ]

                self.registro_extraidas.add(identificador)
                return caracteristicas

        except Exception as e:
            print(
                f"\n[!] Error extrayendo características de {imagen_path}: {str(e)}")
            return None

    def extraer_de_directorio(self, directorio_raiz, archivo_salida):
        self._cargar_registro_extraidas(archivo_salida)

        modo = 'a' if os.path.exists(archivo_salida) else 'w'
        with open(archivo_salida, modo, newline='') as csvfile:
            writer = csv.writer(csvfile)

            if modo == 'w':
                writer.writerow(self.cabeceras)

            contador = 0
            total_archivos = 0

            # Contar total de archivos para progreso
            for clase in os.listdir(directorio_raiz):
                clase_dir = os.path.join(directorio_raiz, clase)
                if os.path.isdir(clase_dir):
                    total_archivos += sum(1 for f in os.listdir(clase_dir)
                                          if f.lower().endswith(('.png', '.jpg', '.jpeg')))

            # Procesar cada clase
            for clase in os.listdir(directorio_raiz):
                clase_dir = os.path.join(directorio_raiz, clase)
                if not os.path.isdir(clase_dir):
                    continue

                for archivo in os.listdir(clase_dir):
                    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path_completo = os.path.join(clase_dir, archivo)
                        caracteristicas = self.extraer_de_imagen(
                            path_completo, clase)

                        if caracteristicas:
                            writer.writerow(caracteristicas)
                            contador += 1
                            print(
                                f"\r[+] Progreso: {contador}/{total_archivos}", end='')

            print(
                f"\n[+] Extracción completada. {contador} nuevas características añadidas")
