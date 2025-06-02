import os
import csv
import numpy as np
import pytesseract
import re
from PIL import Image, ImageOps

def imagen_a_array(imagen):
    return np.array(imagen.convert("L"))

def aplicar_kernel(imagen, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(imagen, ((ph, ph), (pw, pw)), mode='edge')
    resultado = np.zeros_like(imagen, dtype=np.float32)
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            resultado[i, j] = np.sum(region * kernel)
    return resultado

class ExtractorCaracteristicas:
    def __init__(self):
        self.kernel_gaussiano = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        self.kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.kernel_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        os.makedirs("salidas/ocr", exist_ok=True)
        self.resultados_ocr_path = "salidas/ocr/ocr_resultados.txt"
        open(self.resultados_ocr_path, "w", encoding="utf-8").close()

    def _preprocesar_ocr_para_tesseract(self, imagen):
        gris = imagen.convert("L")
        gris = ImageOps.autocontrast(gris)
        arr = np.array(gris)
        umbral = arr.mean()
        binaria = (arr > umbral).astype(np.uint8) * 255
        binaria_img = Image.fromarray(binaria)
        binaria_img = binaria_img.resize((binaria_img.width * 2, binaria_img.height * 2), Image.LANCZOS)
        return binaria_img

    def _realizar_ocr(self, imagen_original, nombre_archivo):
        imagen_proc = self._preprocesar_ocr_para_tesseract(imagen_original)
        config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        texto = pytesseract.image_to_string(imagen_proc, lang='eng', config=config).strip()
        texto_limpio = re.sub(r'[^A-Z0-9 ]', '', texto.upper())
        caracteres = len(texto_limpio)

        with open(self.resultados_ocr_path, "a", encoding="utf-8") as f:
            f.write(f"{nombre_archivo}: {caracteres} → Texto: \"{texto_limpio or 'no'}\"\n")

        return texto_limpio or "no", caracteres

    def _calcular_momentos_hu(self, imagen_bin):
        def momento_central(p, q, img):
            m00 = np.sum(img)
            if m00 == 0: return 0.0
            y, x = np.indices(img.shape)
            cx = np.sum(x * img) / m00
            cy = np.sum(y * img) / m00
            return np.sum(((x - cx)**p) * ((y - cy)**q) * img) / (m00**((p + q)/2 + 1))

        n20 = momento_central(2, 0, imagen_bin)
        n02 = momento_central(0, 2, imagen_bin)
        n11 = momento_central(1, 1, imagen_bin)
        n30 = momento_central(3, 0, imagen_bin)
        n12 = momento_central(1, 2, imagen_bin)
        n21 = momento_central(2, 1, imagen_bin)
        n03 = momento_central(0, 3, imagen_bin)

        return [
            n20 + n02,
            (n20 - n02)**2 + 4 * n11**2,
            (n30 - 3*n12)**2 + (3*n21 - n03)**2,
            (n30 + n12)**2 + (n21 + n03)**2,
            (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) +
            (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2),
            (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) +
            4 * n11 * (n30 + n12)*(n21 + n03),
            (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) -
            (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
        ]

    def _preprocesar_para_caracteristicas(self, imagen):
        array_img = imagen_a_array(imagen)
        suavizada = aplicar_kernel(array_img, self.kernel_gaussiano)
        suavizada_u8 = suavizada.astype(np.uint8)

        hist, _ = np.histogram(suavizada_u8.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        ecualizada = cdf_normalized[suavizada_u8]

        grad_x = aplicar_kernel(ecualizada, self.kernel_sobel_x)
        grad_y = aplicar_kernel(ecualizada, self.kernel_sobel_y)
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        return (magnitud / magnitud.max() * 255).astype(np.uint8)

    def extraer_de_imagen(self, imagen_path, clase):
        try:
            imagen = Image.open(imagen_path).convert("RGB")
            imagen.thumbnail((300, 300))
            nombre_archivo = os.path.basename(imagen_path)
            img_proc = self._preprocesar_para_caracteristicas(imagen)

            ancho, alto = imagen.size
            rel_aspecto = ancho / alto
            intensidad = np.mean(img_proc)
            desviacion = np.std(img_proc)

            hist, _ = np.histogram(img_proc.flatten(), 256, [0, 256])
            hist = hist / hist.sum()
            entropia = -np.sum(hist * np.log2(hist + 1e-10))
            uniformidad = np.sum(hist**2)

            contraste = (np.mean(np.abs(np.diff(img_proc, axis=0))) +
                         np.mean(np.abs(np.diff(img_proc, axis=1)))) / 2

            binaria = (img_proc > np.mean(img_proc)).astype(np.uint8)
            momentos = self._calcular_momentos_hu(binaria)
            area = np.sum(binaria)
            perimetro = np.sum(binaria[:, :-1] != binaria[:, 1:]) + \
                        np.sum(binaria[:-1, :] != binaria[1:, :])
            compacidad = (perimetro ** 2) / (4 * np.pi * area) if area else 0

            texto, chars = self._realizar_ocr(imagen, nombre_archivo)

            return [
                clase, nombre_archivo, ancho, alto, rel_aspecto,
                intensidad, desviacion, entropia, uniformidad, contraste,
                *momentos, area, perimetro, compacidad, 0, 0, chars
            ]
        except Exception as e:
            print(f"[!] Error en {imagen_path}: {e}")
            return None

    def extraer_de_directorio(self, directorio_raiz, archivo_salida):
        modo = 'a' if os.path.exists(archivo_salida) else 'w'

        with open(archivo_salida, modo, newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if modo == 'w':
                writer.writerow([
                    'clase', 'nombre_archivo', 'ancho', 'alto', 'relacion_aspecto',
                    'intensidad_promedio', 'desviacion_intensidad',
                    'entropia', 'uniformidad', 'contraste',
                    'hu_momento_1', 'hu_momento_2', 'hu_momento_3',
                    'hu_momento_4', 'hu_momento_5', 'hu_momento_6', 'hu_momento_7',
                    'area_objetos', 'perimetro_promedio', 'compacidad_promedio',
                    'excentricidad_promedio', 'solidez_promedio', 'ocr_caracteres'
                ])

            contador = 0
            total_archivos = sum(
                1 for clase in os.listdir(directorio_raiz)
                for f in os.listdir(os.path.join(directorio_raiz, clase))
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            )

            for clase in os.listdir(directorio_raiz):
                clase_dir = os.path.join(directorio_raiz, clase)
                if not os.path.isdir(clase_dir):
                    continue

                for archivo in os.listdir(clase_dir):
                    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path_completo = os.path.join(clase_dir, archivo)
                        caracteristicas = self.extraer_de_imagen(path_completo, clase)

                        if caracteristicas:
                            writer.writerow(caracteristicas)
                            contador += 1
                            print(f"\r[+] Progreso: {contador}/{total_archivos}", end='')

            print(f"\n[+] Extracción completada. {contador} nuevas características añadidas")
