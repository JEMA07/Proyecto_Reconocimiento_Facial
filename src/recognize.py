import cv2
import pickle
import numpy as np
import face_recognition
import os
from collections import deque, Counter

# =============================
# CARGA DE MODELO / EMBEDDINGS
# =============================
MODEL_PATH = os.path.join("models", "embeddings_mtcnn.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontró el modelo entrenado.")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"Base cargada con {len(known_names)} rostros registrados.")

# =============================
# DETECCIÓN AUTOMÁTICA DE CÁMARA
# =============================
def detectar_camara():
    print("Buscando cámara activa...")
    for i in range(4):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        print(f"Índice {i}: {'OK' if ret else 'No frame'}")
        if ret:
            cv2.imshow(f"Cámara {i}", frame)
            cv2.waitKey(600)
            cv2.destroyAllWindows()
            cap.release()
            return i
        cap.release()
    print("No se detectó cámara disponible.")
    return None

CAM_INDEX = detectar_camara()
if CAM_INDEX is None:
    raise RuntimeError("No se pudo acceder a la cámara.")

video = cv2.VideoCapture(CAM_INDEX)
print("Cámara conectada correctamente. Iniciando reconocimiento facial...")

# =============================
# PARÁMETROS ANTI FALSOS POSITIVOS
# =============================
THRESH = 0.50           # más estricto (0.45–0.55). Sube si hay muchos no-identificados.
MARGIN = 0.07           # diferencia mínima con la segunda mejor distancia
DETECTOR_MODEL = "hog"  # "hog" (CPU) o "cnn" si tu equipo lo soporta

# Votación temporal para confirmar identidad antes de mostrar nombre
VOTES_WINDOW = 7
VOTES_NEED = 5
recent_votes = deque(maxlen=VOTES_WINDOW)

# =============================
# BUCLE PRINCIPAL
# =============================
while True:
    ret, frame = video.read()
    if not ret or frame is None or frame.size == 0:
        print("Frame inválido.")
        break

    # Mejora ligera para cámaras de baja calidad
    frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección
    locations = face_recognition.face_locations(rgb_frame, model=DETECTOR_MODEL)
    print(f"Detecciones: {len(locations)}")

    # Encodings
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    rostros_procesados = False

    for (encoding, location) in zip(encodings, locations):
        distances = face_recognition.face_distance(known_encodings, encoding)

        # 1) Decisión con umbral + margen (segunda mejor)
        if len(distances) == 0:
            candidate = "Sujeto no identificado"
            best_dist = None
        else:
            order = np.argsort(distances)
            best_idx = int(order[0])
            best_dist = float(distances[best_idx])
            second_best = float(distances[order[1]]) if len(order) > 1 else 1.0

            if (best_dist <= THRESH) and ((second_best - best_dist) >= MARGIN):
                candidate = known_names[best_idx]
            else:
                candidate = "Sujeto no identificado"

        # 2) Votación temporal (N de M frames)
        recent_votes.append(candidate)
        counts = Counter(recent_votes)
        final_name, votes = counts.most_common(1)[0]

        if final_name != "Sujeto no identificado" and votes >= VOTES_NEED:
            shown_name = final_name
            color = (0, 255, 0)   # Verde confirmado
        else:
            shown_name = "Sujeto no identificado"
            color = (0, 0, 255)   # Rojo hasta confirmar

        # 3) Dibujo
        top, right, bottom, left = location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, shown_name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # (Opcional) mostrar distancia del mejor match para calibrar
        if 'best_dist' in locals() and best_dist is not None:
            cv2.putText(frame, f"{best_dist:.3f}", (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rostros_procesados = True

    if not rostros_procesados:
        cv2.putText(frame, "No se detecta ningún rostro", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Reconocimiento Facial Universal", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Reconocimiento finalizado.")
import cv2
import pickle
import numpy as np
import face_recognition
import os
from collections import deque, Counter

# =============================
# CARGA DE MODELO / EMBEDDINGS
# =============================
MODEL_PATH = os.path.join("models", "embeddings_mtcnn.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontró el modelo entrenado.")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"Base cargada con {len(known_names)} rostros registrados.")

# =============================
# DETECCIÓN AUTOMÁTICA DE CÁMARA
# =============================
def detectar_camara():
    print("Buscando cámara activa...")
    for i in range(4):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        print(f"Índice {i}: {'OK' if ret else 'No frame'}")
        if ret:
            cv2.imshow(f"Cámara {i}", frame)
            cv2.waitKey(600)
            cv2.destroyAllWindows()
            cap.release()
            return i
        cap.release()
    print("No se detectó cámara disponible.")
    return None

CAM_INDEX = detectar_camara()
if CAM_INDEX is None:
    raise RuntimeError("No se pudo acceder a la cámara.")

video = cv2.VideoCapture(CAM_INDEX)
print("Cámara conectada correctamente. Iniciando reconocimiento facial...")

# =============================
# PARÁMETROS ANTI FALSOS POSITIVOS
# =============================
THRESH = 0.50           # más estricto (0.45–0.55). Sube si hay muchos no-identificados.
MARGIN = 0.07           # diferencia mínima con la segunda mejor distancia
DETECTOR_MODEL = "hog"  # "hog" (CPU) o "cnn" si tu equipo lo soporta

# Votación temporal para confirmar identidad antes de mostrar nombre
VOTES_WINDOW = 7
VOTES_NEED = 5
recent_votes = deque(maxlen=VOTES_WINDOW)

# =============================
# BUCLE PRINCIPAL
# =============================
while True:
    ret, frame = video.read()
    if not ret or frame is None or frame.size == 0:
        print("Frame inválido.")
        break

    # Mejora ligera para cámaras de baja calidad
    frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección
    locations = face_recognition.face_locations(rgb_frame, model=DETECTOR_MODEL)
    print(f"Detecciones: {len(locations)}")

    # Encodings
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    rostros_procesados = False

    for (encoding, location) in zip(encodings, locations):
        distances = face_recognition.face_distance(known_encodings, encoding)

        # 1) Decisión con umbral + margen (segunda mejor)
        if len(distances) == 0:
            candidate = "Sujeto no identificado"
            best_dist = None
        else:
            order = np.argsort(distances)
            best_idx = int(order[0])
            best_dist = float(distances[best_idx])
            second_best = float(distances[order[1]]) if len(order) > 1 else 1.0

            if (best_dist <= THRESH) and ((second_best - best_dist) >= MARGIN):
                candidate = known_names[best_idx]
            else:
                candidate = "Sujeto no identificado"

        # 2) Votación temporal (N de M frames)
        recent_votes.append(candidate)
        counts = Counter(recent_votes)
        final_name, votes = counts.most_common(1)[0]

        if final_name != "Sujeto no identificado" and votes >= VOTES_NEED:
            shown_name = final_name
            color = (0, 255, 0)   # Verde confirmado
        else:
            shown_name = "Sujeto no identificado"
            color = (0, 0, 255)   # Rojo hasta confirmar

        # 3) Dibujo
        top, right, bottom, left = location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, shown_name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # (Opcional) mostrar distancia del mejor match para calibrar
        if 'best_dist' in locals() and best_dist is not None:
            cv2.putText(frame, f"{best_dist:.3f}", (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rostros_procesados = True

    if not rostros_procesados:
        cv2.putText(frame, "No se detecta ningún rostro", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Reconocimiento Facial Universal", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Reconocimiento finalizado.")
