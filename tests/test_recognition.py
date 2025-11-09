import cv2
import face_recognition
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from mtcnn.mtcnn import MTCNN
import csv

# === Configuración ===
BASE_DIR = Path(__file__).resolve().parents[1]  # tests/ → raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "embeddings_mtcnn.pkl"
CSV_PATH = BASE_DIR / "data" / "estudiantes.csv"
THRESHOLD = 0.6
OUT_DIR = BASE_DIR / "data" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Cargar modelo entrenado ===
print(" Cargando modelo entrenado...")
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
known_encodings = np.asarray(data["encodings"])
known_names = list(data["names"])
print(f"Modelo cargado con {len(known_names)} rostros registrados.\n")

# === Cargar información de los estudiantes ===
print("cv Cargando base de datos de estudiantes...")
students_info = {}
with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        codigo = row['codigo']
        students_info[codigo] = {
            'nombre': row.get('nombre', ''),
            'apellido': row.get('apellido', ''),
            'grado': row.get('grado', ''),
            'ruta': row.get('ruta_carpeta', '')
        }
print(f" Se cargaron {len(students_info)} registros de estudiantes.\n")

# === Imagen de prueba ===
default_image = BASE_DIR / "data" / "dataset" / "test.jpg"
user_in = input(" Ruta de imagen (ENTER usa data/dataset/test.jpg): ").strip()
test_image = Path(user_in) if user_in else default_image
if not test_image.is_absolute():
    test_image = BASE_DIR / test_image
if not test_image.exists():
    raise FileNotFoundError(f" No se encontró la imagen: {test_image}")

# === Cargar imagen ===
image = cv2.imread(str(test_image))
if image is None:
    raise RuntimeError(f"No se pudo leer la imagen: {test_image}")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Detección de rostro ===
print(" Detectando rostros...")
detector = MTCNN()
faces = detector.detect_faces(rgb_image)

boxes_trbl = []
for face in faces:
    x, y, w, h = face['box']
    top = max(0, y); left = max(0, x)
    bottom = top + max(0, h); right = left + max(0, w)
    boxes_trbl.append((top, right, bottom, left))

if len(boxes_trbl) == 0:
    print("No se detectaron rostros en la imagen.")
else:
    encodings = face_recognition.face_encodings(rgb_image, boxes_trbl)
    for (top, right, bottom, left), encoding in zip(boxes_trbl, encodings):
        if encoding is None or encoding.shape[0] == 0:
            print("No se pudo obtener encoding del rostro detectado.")
            continue

        distances = face_recognition.face_distance(known_encodings, encoding)
        min_distance = float(np.min(distances)) if distances.size else 1.0
        best_match_index = int(np.argmin(distances)) if distances.size else -1

        name = "Desconocido"; extra_info = {}; codigo = ""
        if distances.size and min_distance < THRESHOLD:
            name = known_names[best_match_index]
            codigo = name.split('_')[0] if '_' in name else name
            extra_info = students_info.get(codigo, {})

        # Etiqueta con nombre + grado
        nombre_mostrar = extra_info.get('nombre', name)
        grado_mostrar = extra_info.get('grado', '')
        etiqueta = f"{nombre_mostrar} - {grado_mostrar}" if grado_mostrar else nombre_mostrar

        # Dibujar
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, etiqueta, (left, max(10, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# === Guardar resultado (incluye código y grado si existen) ===
sufijo = ""
if 'codigo' in locals() and codigo:
    sufijo += f"_{codigo}"
if 'grado_mostrar' in locals() and grado_mostrar:
    sufijo += f"_{grado_mostrar}"

output_path = OUT_DIR / (test_image.stem + f"{sufijo}_resultado.jpg")
cv2.imwrite(str(output_path), image)
print(f"\n Resultado guardado en: {output_path}")
