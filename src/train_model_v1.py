# ·train_model_v1.py· 
import os
import csv
import cv2
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import face_recognition
# Si prefieres MTCNN, activa USE_MTCNN=True
# from mtcnn.mtcnn import MTCNN
USE_MTCNN = False  # face_recognition.face_locations por defecto

# ---------------- Configuración por defecto ----------------
# Si este archivo está en src/, usa parents[1] para apuntar a la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "embeddings_mtcnn.pkl"
CSV_PATH = BASE_DIR / "data" / "estudiantes.csv"
DATASET_DIR = BASE_DIR / "data" / "dataset"
OUT_IMG_DIR = BASE_DIR / "data" / "out" / "recognitions"
OUT_CSV_PATH = BASE_DIR / "data" / "out" / "results.csv"
THRESHOLD = 0.6  # más bajo = más estricto

def ensure_dirs(out_img_dir: Path, out_csv_path: Path):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

def load_model(model_path: Path):
    print(" Cargando modelo entrenado...")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    known_encodings = np.asarray(data["encodings"])
    known_names = list(data["names"])
    print(f" Modelo cargado con {len(known_names)} rostros registrados.\n")
    return known_encodings, known_names

def load_students(csv_path: Path):
    print(" Cargando base de datos de estudiantes...")
    students_info = {}
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            codigo = row["codigo"]
            students_info[codigo] = {
                "nombre": row.get("nombre", ""),
                "apellido": row.get("apellido", ""),
                "grado": row.get("grado", ""),
                "ruta": row.get("ruta_carpeta", ""),
            }
    print(f" Se cargaron {len(students_info)} registros de estudiantes.\n")
    return students_info

def draw_and_label(image, box, label):
    # box: (top, right, bottom, left)
    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, label, (left, max(10, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def detect_boxes(rgb_image):
    if USE_MTCNN:
        from mtcnn.mtcnn import MTCNN
        detector = MTCNN()
        faces = detector.detect_faces(rgb_image)
        boxes = []
        for f in faces:
            x, y, w, h = f["box"]
            top = max(0, y)
            left = max(0, x)
            bottom = top + max(0, h)
            right = left + max(0, w)
            boxes.append((top, right, bottom, left))
        return boxes
    else:
        return face_recognition.face_locations(rgb_image, model="hog")

def process_image(img_path: Path, known_encodings, known_names, students_info, threshold: float):
    image = cv2.imread(str(img_path))
    if image is None:
        return [], None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = detect_boxes(rgb)
    encs = face_recognition.face_encodings(rgb, boxes)

    results_for_image = []
    for box, enc in zip(boxes, encs):
        if enc is None or enc.shape[0] == 0:
            continue

        dists = face_recognition.face_distance(known_encodings, enc)
        if dists.size == 0:
            name = "Desconocido"
            min_d = 1.0
            codigo = ""
            extra = {}
        else:
            idx = int(np.argmin(dists))
            min_d = float(dists[idx])
            if min_d < threshold:
                name = known_names[idx]
                codigo = name.split("_")[0] if "_" in name else name
                extra = students_info.get(codigo, {})
            else:
                name = "Desconocido"
                codigo = ""
                extra = {}

        label = extra.get("nombre", name)
        draw_and_label(image, box, label)
         # >>> AQUI va la etiqueta con grado <<<
        nombre_mostrar = extra.get("nombre", name)
        grado_mostrar = extra.get("grado", "")
        etiqueta = f"{nombre_mostrar} - {grado_mostrar}" if grado_mostrar else nombre_mostrar

        draw_and_label(image, box, etiqueta)

        results_for_image.append({
            "file": str(img_path),
            "name": name,
            "codigo": codigo,
            "nombre": extra.get("nombre", ""),
            "apellido": extra.get("apellido", ""),
            "grado": extra.get("grado", ""),
            "distancia": round(min_d, 3),
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })

    return results_for_image, image

def iter_images(dataset_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            if Path(fn).suffix.lower() in exts:
                yield Path(root) / fn

def run(dataset_dir: Path, model_path: Path, students_csv: Path,
        out_dir: Path, out_csv: Path, threshold: float):
    ensure_dirs(out_dir, out_csv)
    known_encodings, known_names = load_model(model_path)
    students_info = load_students(students_csv)

    all_results = []
    count_imgs = 0
    for img_path in iter_images(dataset_dir):
        count_imgs += 1
        results, annotated = process_image(img_path, known_encodings, known_names, students_info, threshold)
        all_results.extend(results)
        if annotated is not None:
            out_path = out_dir / (img_path.stem + "_resultado.jpg")
            cv2.imwrite(str(out_path), annotated)
        if count_imgs % 50 == 0:
            print(f"🧾 Procesadas {count_imgs} imágenes...")

    fieldnames = ["file", "name", "codigo", "nombre", "apellido", "grado", "distancia", "timestamp"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f" Imágenes procesadas: {count_imgs}. Resultados en {out_csv}. Salidas en {out_dir}")

def build_parser():
    p = argparse.ArgumentParser(description="Reconocimiento por lotes desde data/dataset.")
    p.add_argument("--dataset", type=str, default=str(DATASET_DIR), help="Carpeta dataset.")
    p.add_argument("--model", type=str, default=str(MODEL_PATH), help="Pickle con embeddings.")
    p.add_argument("--students", type=str, default=str(CSV_PATH), help="CSV de estudiantes.")
    p.add_argument("--out_dir", type=str, default=str(OUT_IMG_DIR), help="Salida de imágenes anotadas.")
    p.add_argument("--out_csv", type=str, default=str(OUT_CSV_PATH), help="CSV de resultados.")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Umbral de aceptación.")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    run(
        dataset_dir=Path(args.dataset),
        model_path=Path(args.model),
        students_csv=Path(args.students),
        out_dir=Path(args.out_dir),
        out_csv=Path(args.out_csv),
        threshold=float(args.threshold),
    )

if __name__ == "__main__":
    main()
