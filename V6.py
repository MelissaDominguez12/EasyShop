import cv2
import numpy as np
import pandas as pd
from skimage import measure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os
import random
from tkinter import Tk, Toplevel, Label, Button, filedialog, PhotoImage
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# --- Paso 1: Configuración de las rutas ---
image_paths = {
    0: ['dataset/pritt1.png', 'dataset/pritt2.png', 'dataset/pritt3.png', 'dataset/pritt4.png', 'dataset/pritt5.png'],
    1: ['dataset/lapiz1.png', 'dataset/lapiz2.png', 'dataset/lapiz3.png', 'dataset/lapiz4.png', 'dataset/lapiz5.png'],
    2: ['dataset/clips1.png', 'dataset/clips2.png', 'dataset/clips3.png', 'dataset/clips4.png', 'dataset/clips5.png'],
    3: ['dataset/tachuelas1.png', 'dataset/tachuelas2.png', 'dataset/tachuelas3.png', 'dataset/tachuelas4.png', 'dataset/tachuelas5.png'],
    4: ['dataset/plumas1.png', 'dataset/plumas2.png', 'dataset/plumas3.png', 'dataset/plumas4.png', 'dataset/plumas5.png']
}

label_names = {
    0: 'Pritt',
    1: 'Lápiz',
    2: 'Clips',
    3: 'Tachuelas',
    4: 'Plumas'
}

# --- Paso 2: Funciones de procesamiento de imágenes ---
def extract_features(image_path, label):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    labeled_img = measure.label(binary)
    regions = measure.regionprops(labeled_img)

    features = []
    for i, region in enumerate(regions):
        if region.area > 50:  # Filtrar ruido
            features.append([
                f"{os.path.basename(image_path).split('.')[0]}_OB{i+1}",
                region.area,
                region.perimeter,
                region.eccentricity,
                region.extent,
                region.solidity,
                region.major_axis_length,
                region.minor_axis_length,
                region.convex_area,
                region.equivalent_diameter,
                label
            ])
    return features

def encontrar_objetos_conectividad_8(binary_image):
    labeled_image = np.zeros_like(binary_image, dtype=int)
    label = 1

    def flood_fill(x, y, label):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if binary_image[cx, cy] == 255 and labeled_image[cx, cy] == 0:
                labeled_image[cx, cy] = label
                for nx, ny in [
                    (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1),
                    (cx-1, cy-1), (cx-1, cy+1), (cx+1, cy-1), (cx+1, cy+1)
                ]:
                    if 0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1]:
                        stack.append((nx, ny))

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 255 and labeled_image[i, j] == 0:
                flood_fill(i, j, label)
                label += 1

    return labeled_image, label - 1

# --- Paso 3: Construcción del dataset ---
data = []

for label, paths in image_paths.items():
    for path in paths:
        if os.path.exists(path):
            features = extract_features(path, label)
            data.extend(features)

pruebas_path = 'dataset/pruebas'
if os.path.exists(pruebas_path):
    for prueba_img in os.listdir(pruebas_path):
        prueba_img_path = os.path.join(pruebas_path, prueba_img)
        if os.path.isfile(prueba_img_path):
            features = extract_features(prueba_img_path, label=None)
            data.extend(features)

columns = ['Object', 'Area', 'Perimeter', 'Eccentricity', 'Extent', 'Solidity', 'MajorAxisLength', 'MinorAxisLength', 'ConvexArea', 'EquivalentDiameter', 'Label']
df = pd.DataFrame(data, columns=columns)

df.to_csv("object_features.csv", index=False)

X = df.drop(['Object', 'Label'], axis=1)
X = X.astype(float)
y = df['Label']

X = X[~y.isnull()]
y = y[~y.isnull()]

selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

def classify_mixed_image(image_path):
    features = extract_features(image_path, label=None)
    if not features:
        print("No se pudieron extraer características significativas de la imagen.")
        return []

    nop_folder = 'Prueba2'
    if os.path.commonpath([os.path.abspath(image_path), os.path.abspath(nop_folder)]) == os.path.abspath(nop_folder):
        print("La imagen no se encuentra en el contexto y no se puede clasificar.")
        return []
    
    features_df = pd.DataFrame(features, columns=['Object'] + list(X.columns) + ['Label'])
    features_df = features_df[list(X.columns)]
    scaled_features = scaler.transform(selector.transform(features_df))
    predictions = knn.predict(scaled_features)

    probabilities = knn.predict_proba(scaled_features)
    max_prob = probabilities.max(axis=1)
    confident_predictions = [pred for pred, prob in zip(predictions, max_prob) if prob > 0.8]

    if not confident_predictions:
        print("Objeto no reconocido.")
        return []

    class_names = list(set(label_names[pred] for pred in confident_predictions))
    if len(class_names) > 2:
        class_names = class_names[:2]  # Limitar a dos clases
    return class_names

def display_image_with_classes(image_path, classes):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure("Imagen seleccionada")
    if classes:
        plt.title(f"Clases detectadas: {', '.join(classes)}")
    else:
        plt.title("Objeto no reconocido")
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def display_related_images(selected_image_path, class_names):
    if not class_names:
        print("No hay imágenes relacionadas que mostrar.")
        return

    related_images = []

    # Procesar la imagen seleccionada
    image = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    labeled_img = measure.label(binary)
    regions = measure.regionprops(labeled_img)
    selected_image_objects = sum(1 for region in regions if region.area > 50)  # Contar objetos significativos

    # Determinar si la imagen está en la carpeta 'prueba'
    is_in_prueba_folder = "prueba" in os.path.abspath(selected_image_path).lower()

    if not is_in_prueba_folder:
        # Añadir la imagen seleccionada como referencia solo si no está en 'prueba'
        related_images.append((selected_image_path, selected_image_objects, 100, "Imagen seleccionada"))

    # Procesar las imágenes relacionadas
    for class_name in class_names:
        class_label = [key for key, value in label_names.items() if value == class_name][0]
        class_images = image_paths[class_label]

        for image_path in class_images:
            if image_path != selected_image_path and os.path.exists(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
                labeled_img = measure.label(binary)
                regions = measure.regionprops(labeled_img)
                num_objects = sum(1 for region in regions if region.area > 50)  # Contar objetos significativos

                # Calcular el porcentaje de similitud basado en el número de objetos
                similarity = 100 - abs(selected_image_objects - num_objects) / max(selected_image_objects, 1) * 100

                # Guardar información de la relación con énfasis en la coincidencia exacta de objetos
                match_type = "Coincidencia exacta" if num_objects == selected_image_objects else "Relacionada"
                related_images.append((image_path, num_objects, similarity, match_type))

    # Ordenar primero por coincidencia exacta y luego por porcentaje de similitud
    related_images.sort(key=lambda x: (x[3] == "Coincidencia exacta", x[2]), reverse=True)

    # Mostrar las imágenes relacionadas
    plt.figure("Imágenes Relacionadas")
    for i, (image_path, num_objects, similarity, match_type) in enumerate(related_images[:5]):
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 5, i + 1)
        plt.imshow(image_rgb)
        plt.title(f"Imagen {i + 1}\n{similarity:.1f}%")
        plt.axis('off')
    plt.show()



def main():
    def open_file():
        file_path = filedialog.askopenfilename(title="Seleccione una imagen", filetypes=[("Archivos de imagen", "*.jpeg *.jpg *.png *.bmp")])
        if file_path:
            predictions = classify_mixed_image(file_path)
            if predictions:
                print(f"Clases presentes en la imagen seleccionada: {predictions}")
            else:
                print("La imagen seleccionada no contiene objetos reconocibles.")
            display_image_with_classes(file_path, predictions)
            display_related_images(file_path, predictions)

    def open_combined_view():
        file_path = filedialog.askopenfilename(title="Seleccione una imagen", filetypes=[("Archivos de imagen", "*.jpeg *.jpg *.png *.bmp")])
        if file_path:
            predictions = classify_mixed_image(file_path)
            display_image_with_classes(file_path, predictions)
            display_related_images(file_path, predictions)

    root = Tk()
    root.title("Clasificador de Imágenes")
    root.geometry("800x600")
    root.configure(bg="white")

    welcome_label = Label(root, text="Clasificador de Imágenes", bg="white", font=("Arial", 24))
    welcome_label.pack(pady=20)

    select_button = Button(root, text="Ver Imagen y Relacionadas", command=open_combined_view, bg="blue", fg="white", font=("Arial", 14))
    select_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()


