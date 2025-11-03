# Entrenamiento del modelo facial
import os
import cv2
import face_recognition
import pickle
import numpy as np

# Rutas base
DATASET_DIR = os.path.join("data", "dataset")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.pkl")

# Inicializar listas
known_encodings = []
known_names = []

print("Iniciando proceso de entrenamiento facial...\n")

# Recorrer cada carpeta de estudiante
for student_folder in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, student_folder)
    if not os.path.isdir(student_path):
        continue

    student_id = student_folder.split("_")[0]
    student_name = "_".join(student_folder.split("_")[1:])
    print(f"Procesando estudiante: {student_name} (ID: {student_id})")

    # Recorrer las imágenes del estudiante
    for image_file in os.listdir(student_path):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(student_path, image_file)
        image = face_recognition.load_image_file(image_path)

        # Detección de rostro
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            print(f"No se detectó rostro en: {image_file}")
            continue

        # Obtener el embedding (vector del rostro)
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_encodings.append(encoding)
        known_names.append(f"{student_id}_{student_name}")

print("\nGuardando modelo entrenado...")

# Guardar embeddings
data = {"encodings": known_encodings, "names": known_names}
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"Entrenamiento completado. Archivo guardado en: {EMBEDDINGS_FILE}")
