import os
import numpy as np
from PIL import Image

# Cargar plantillas desde carpetas organizadas por clase
def cargar_plantillas(path='dataset', tamaño=(20, 20)):
    plantillas = []
    clases = []

    for clase in os.listdir(path):
        clase_path = os.path.join(path, clase)
        if os.path.isdir(clase_path):
            for archivo in os.listdir(clase_path):
                img_path = os.path.join(clase_path, archivo)
                try:
                    img = Image.open(img_path).convert("L").resize(tamaño)
                except Exception:
                    continue
                img_array = np.array(img)
                img_array = img_array / 255.0
                plantillas.append(img_array.flatten())
                clases.append(clase)

    return np.array(plantillas), np.array(clases)

# Reconocer una imagen comparando contra las plantillas por distancia euclidiana
def reconocer_por_distancia(imagen_path, plantillas, clases, tamaño=(20, 20)):
    try:
        img = Image.open(imagen_path).convert("L").resize(tamaño)
    except Exception:
        return "error"
    img_array = np.array(img) / 255.0
    img_flat = img_array.flatten()
    distancias = np.linalg.norm(plantillas - img_flat, axis=1)
    indice_minimo = np.argmin(distancias)
    return clases[indice_minimo]

# Evaluar el modelo sobre un conjunto de prueba
def evaluar_modelo(directorio_pruebas, plantillas, clases, tamaño=(20, 20)):
    total = 0
    correctas = 0
    for clase in os.listdir(directorio_pruebas):
        clase_path = os.path.join(directorio_pruebas, clase)
        if not os.path.isdir(clase_path):
            continue
        for archivo in os.listdir(clase_path):
            img_path = os.path.join(clase_path, archivo)
            pred = reconocer_por_distancia(img_path, plantillas, clases, tamaño)
            if pred == clase:
                correctas += 1
            total += 1
            print(f"{archivo}: esperado={clase}, predicho={pred}")
    if total > 0:
        print(f"[+] Precisión: {correctas}/{total} = {correctas / total:.2%}")

# Segmentar caracteres desde una imagen completa en escala de grises
def segmentar_letras(imagen, umbral=20):
    img = imagen.convert("L")
    arr = np.array(img)
    suma_columnas = np.sum(arr < 128, axis=0)
    letras = []
    inicio = None
    for i, valor in enumerate(suma_columnas):
        if valor > umbral and inicio is None:
            inicio = i
        elif valor <= umbral and inicio is not None:
            if i - inicio > 2:
                letra_arr = arr[:, inicio:i]
                letras.append(Image.fromarray(letra_arr))
            inicio = None
    return letras

# Ejemplo de uso directo
if __name__ == "__main__":
    plantillas, clases = cargar_plantillas("dataset")
    resultado = reconocer_por_distancia("test.png", plantillas, clases)
    print("Letra reconocida:", resultado)
