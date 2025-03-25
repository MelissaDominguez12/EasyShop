import os
import numpy as np
from PIL import Image


input_folder = "Dataset/"
output_folder = "Dataset_procesado/"

# Definir clases
clases = ['crema', 'jalapenos', 'leche', 'maizena', 'pelon']


def to_grayscale(image):
    image = image.convert('RGB')

    width, height = image.size
    pixels = image.load()

    gray_image = Image.new('L', (width, height))
    gray_pixels = gray_image.load()

    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((i, j))
            gray_pixels[i, j] = int((r + g + b) / 3)

    return gray_image


def apply_threshold(image, threshold=128):
    width, height = image.size
    pixels = image.load()

    for i in range(width):
        for j in range(height):
            pixel = pixels[i, j]
            if pixel < threshold:
                pixels[i, j] = 0
            else:
                pixels[i, j] = 255

    return image


def flood_fill(image, x, y, visited, width, height):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()

        if cx < 0 or cx >= width or cy < 0 or cy >= height or (cx, cy) in visited:
            continue

        visited.add((cx, cy))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in neighbors:
            stack.append((cx + dx, cy + dy))


for clase in clases:
    input_class_path = os.path.join(input_folder, clase)
    output_class_path = os.path.join(output_folder, clase + '_pros')

    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)

    for filename in os.listdir(input_class_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(input_class_path, filename)
            image = Image.open(img_path)

            gray_image = to_grayscale(image)

            binary_image = apply_threshold(gray_image, threshold=128)

            width, height = binary_image.size
            pixels = binary_image.load()
            visited = set()

            for i in range(width):
                for j in range(height):
                    if pixels[i, j] == 255 and (i, j) not in visited:
                        flood_fill(binary_image, i, j, visited, width, height)

            output_path = os.path.join(
                output_class_path, f"{os.path.splitext(filename)[0]}_pros.png")
            binary_image.save(output_path)

print("Procesamiento completado.")
