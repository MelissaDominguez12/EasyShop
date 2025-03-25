import cv2
import time
import os

# Crear la carpeta si no existe
if not os.path.exists('./Dataset/imagenes'):
    os.makedirs('./Dataset/imagenes')

# Abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# Captura de imágenes cada 0.5 segundos
counter = 0
while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    # Guardar la imagen
    file_name = f"./Dataset/imagenes/imagen_{counter}.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Imagen guardada: {file_name}")

    # Incrementar el contador y esperar 0.5 segundos
    counter += 1
    time.sleep(1)

# Liberar la cámara
cap.release()
cv2.destroyAllWindows()
