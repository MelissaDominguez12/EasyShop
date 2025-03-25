import cv2
import time
import os
import subprocess
import threading

# Crear la carpeta si no existe
if not os.path.exists('./Dataset/imagenes'):
    os.makedirs('./Dataset/imagenes')

capturando = False  # Variable de estado para saber si está capturando


def capturar_imagenes():
    global capturando  # Usamos la variable global
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    # Captura de imágenes cada 0.5 segundos
    counter = 0
    while capturando:
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
        time.sleep(0.5)

    cap.release()  # Liberar la cámara, no es necesario cv2.destroyAllWindows()


def ejecutar_procesamiento():
    # Ejecuta el script procesamiento.py
    print("Ejecutando el script de procesamiento...")
    # Asumiendo que el archivo se llama procesamiento.py
    subprocess.run(["python", "procesamiento.py"])


def main():
    global capturando
    print("Presiona Enter para comenzar a capturar imágenes.")
    while True:
        input()  # Espera por una tecla Enter para comenzar o detener
        if not capturando:
            print("Captura iniciada.")
            capturando = True
            # Usamos threading para que la captura no bloquee la interfaz
            captura_thread = threading.Thread(target=capturar_imagenes)
            captura_thread.start()
        else:
            print("Captura detenida.")
            capturando = False
            captura_thread.join()  # Espera que termine el hilo antes de continuar
            print("Captura detenida. Ejecutando procesamiento de imágenes...")
            ejecutar_procesamiento()
            print("Procesamiento completado.")


if __name__ == "__main__":
    main()
