# src/recognize_mtcnn.py
import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
import face_recognition
import os

# =============================
# CONFIGURACIONES INICIALES
# =============================
MODEL_PATH = os.path.join("models", "embeddings_mtcnn.pkl")

# Cargar modelo entrenado
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo entrenado en {MODEL_PATH}")

print("Cargando modelo de embeddings...")
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f" Modelo cargado con {len(known_names)} rostros registrados.")

# Inicializar detector MTCNN
detector = MTCNN()

# Inicializar cámara (0 = cámara principal)
video = cv2.VideoCapture(1)
if not video.isOpened():
    raise RuntimeError(" No se pudo acceder a la cámara.")

print("🎥 Iniciando reconocimiento facial en tiempo real (presiona 'q' para salir)...")

# =============================
# BUCLE PRINCIPAL
# =============================
while True:
    ret, frame = video.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    for det in detections:
        x, y, w, h = det['box']
        x, y = abs(x), abs(y)
        face_crop = rgb_frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        # Obtener encoding del rostro detectado
        encodings = face_recognition.face_encodings(face_crop)
        if len(encodings) == 0:
            continue

        face_encoding = encodings[0]

        # Comparar con el modelo entrenado
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Desconocido"

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            name = known_names[match_index]

        # Dibujar el recuadro y el nombre
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar imagen
    cv2.imshow("Reconocimiento Facial - MTCNN", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# FINALIZAR
# =============================
video.release()
cv2.destroyAllWindows()
print(" Reconocimiento finalizado.")
