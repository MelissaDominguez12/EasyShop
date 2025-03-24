import os
import numpy as np
from PIL import Image  # Esto es solo para simular la lectura de la imagen. Es un método básico para abrir.

# Asumimos que las imágenes están en formato PNG o JPEG
# Cambiar 'ruta_a_imagen' con las rutas de tus archivos.
input_folder = "Dataset/"
output_folder = "Dataset_procesado/"

# Definir clases
clases = ['crema', 'jalapenos', 'leche', 'maizena', 'pelon']

# Filtro básico: Conversión a escala de grises
from PIL import Image

# Filtro básico: Conversión a escala de grises
def to_grayscale(image):
    # Convertir a RGB si no es
    image = image.convert('RGB')  # Esto se asegura de que la imagen esté en formato RGB
    
    width, height = image.size
    pixels = image.load()
    
    gray_image = Image.new('L', (width, height))  # 'L' es para imagen en escala de grises
    gray_pixels = gray_image.load()
    
    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((i, j))
            gray_pixels[i, j] = int((r + g + b) / 3)  # Promedio de los canales RGB
    
    return gray_image


# Umbralizar la imagen para segmentar
def apply_threshold(image, threshold=128):
    width, height = image.size
    pixels = image.load()
    
    for i in range(width):
        for j in range(height):
            pixel = pixels[i, j]
            if pixel < threshold:
                pixels[i, j] = 0  # Negro
            else:
                pixels[i, j] = 255  # Blanco
    
    return image

# Conectividad 8: Flood Fill (relleno de píxeles conectados)
def flood_fill(image, x, y, visited, width, height):
    stack = [(x, y)]  # Usamos una pila para evitar la recursión
    while stack:
        cx, cy = stack.pop()
        
        # Verifica si el píxel está dentro de los límites y no ha sido visitado
        if cx < 0 or cx >= width or cy < 0 or cy >= height or (cx, cy) in visited:
            continue
        
        visited.add((cx, cy))  # Marca el píxel como visitado
        
        # Verifica los 8 vecinos
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in neighbors:
            stack.append((cx + dx, cy + dy))


# Procesar todas las imágenes de cada clase
for clase in clases:
    input_class_path = os.path.join(input_folder, clase)
    output_class_path = os.path.join(output_folder, clase + '_pros')
    
    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)
    
    for filename in os.listdir(input_class_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Leer imagen
            img_path = os.path.join(input_class_path, filename)
            image = Image.open(img_path)
            
            # Convertir a escala de grises
            gray_image = to_grayscale(image)
            
            # Aplicar umbralización
            binary_image = apply_threshold(gray_image, threshold=128)
            
            # Crear imagen procesada con conectividad 8
            width, height = binary_image.size
            pixels = binary_image.load()
            visited = set()
            
            for i in range(width):
                for j in range(height):
                    if pixels[i, j] == 255 and (i, j) not in visited:
                        flood_fill(binary_image, i, j, visited, width, height)
            
            # Guardar imagen procesada
            output_path = os.path.join(output_class_path, f"{os.path.splitext(filename)[0]}_pros.png")
            binary_image.save(output_path)

print("Procesamiento completado.")
