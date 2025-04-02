import os
import numpy as np


def crear_directorios():
    """Crea todos los directorios necesarios para el proyecto"""
    directorios = [
        'datasets',
        'capturas',
        'salidas/dataset',
        'salidas/camara',
        'features'
    ]
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)


def imagen_a_array(imagen):
    """Convierte una imagen PIL a array numpy manualmente"""
    ancho, alto = imagen.size
    array = np.zeros((alto, ancho), dtype=np.uint8)

    for y in range(alto):
        for x in range(ancho):
            pixel = imagen.getpixel((x, y))
            if isinstance(pixel, int):  # Escala de grises
                array[y, x] = pixel
            else:  # RGB
                r, g, b = pixel[:3]
                array[y, x] = int(0.299 * r + 0.587 * g + 0.114 * b)
    return array


def aplicar_kernel(imagen, kernel):
    """Aplica convolución manual con un kernel"""
    alto, ancho = imagen.shape
    k_alt, k_anc = kernel.shape
    pad_alt = k_alt // 2
    pad_anc = k_anc // 2

    # Añadir padding
    padded = np.pad(imagen, ((pad_alt, pad_alt),
                    (pad_anc, pad_anc)), mode='constant')
    resultado = np.zeros_like(imagen)

    for y in range(alto):
        for x in range(ancho):
            region = padded[y:y+k_alt, x:x+k_anc]
            resultado[y, x] = np.sum(region * kernel)

    return resultado
