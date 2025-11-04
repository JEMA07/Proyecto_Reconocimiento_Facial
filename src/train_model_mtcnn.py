# src/train_model_mtcnn.py
import os
import cv2
import pickle
import numpy as np
import face_recognition
from mtcnn.mtcnn import MTCNN

# === CONFIGURACIONES ===
DATASET_DIR = os.path.join("data", "dataset")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings_mtcnn.pkl")

# Inicializamos el detector MTCNN
detector = MTCNN()

known_encodings = []
known_names = []

print("🚀 Iniciando proceso de entrenamiento facial con MTCNN...\n")

# === RECORRER CADA CARPETA DE ESTUDIANTE ===
for student_folder in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, student_folder)
    if not os.path.isdir(student_path):
        continue

    student_id = student_folder.split("_")[0]
    student_name = "_".join(student_folder.split("_")[1:])
    print(f"Procesando estudiante: {student_name} (ID: {student_id})")

    for image_file in os.listdir(student_path):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(student_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ No se pudo leer la imagen: {image_file}")
            continue

        # Convertir a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # === DETECCIÓN CON MTCNN ===
        faces = detector.detect_faces(image)
        if len(faces) == 0:
            print(f"No se detectó rostro en: {image_file}")
            continue

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            top, right, bottom, left = y, x + w, y + h, x

            # Generar encoding usando face_recognition
            encodings = face_recognition.face_encodings(image, [(top, right, bottom, left)])
            if len(encodings) == 0:
                print(f" No se pudo generar encoding en: {image_file}")
                continue

            known_encodings.append(encodings[0])
            known_names.append(f"{student_id}_{student_name}")

print("\n Guardando modelo entrenado...")

# === GUARDAR EMBEDDINGS ===
data = {"encodings": known_encodings, "names": known_names}
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(data, f)
def main():
    # ... parse args
    global OUT_IMG_DIR, OUT_CSV_PATH
    OUT_IMG_DIR = Path(args.out_dir)
    OUT_CSV_PATH = Path(args.out_csv)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"✅ Entrenamiento completado.")
print(f"📂 Archivo guardado en: {EMBEDDINGS_FILE}")
print(f"👥 Total de rostros entrenados: {len(known_encodings)}")
print(f"🧾 Nombres registrados: {set(known_names)}")
