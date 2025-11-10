import argparse, time, csv, os
from datetime import datetime
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import face_recognition
import pickle

from src.config import LAST_FRAME, EVENTS_CSV, SNAP_DIR

MODEL_PATH = os.path.join("models", "embeddings_mtcnn.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontró el modelo entrenado.")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"Base cargada con {len(known_names)} rostros registrados.")

THRESH = 0.50
MARGIN = 0.07
DETECTOR_MODEL = "hog"

VOTES_WINDOW = 7
VOTES_NEED = 5

def ensure_csv_header():
    if not EVENTS_CSV.exists():
        EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","cam_id","name","codigo","grado","distancia","decision","quality","snapshot_path"])

def append_event(cam_id, name, codigo, grado, distancia, decision, quality, snapshot_path):
    ensure_csv_header()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with EVENTS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, cam_id, name, codigo, grado, distancia, decision, quality, snapshot_path])

def save_frame(frame_bgr):
    LAST_FRAME.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(LAST_FRAME), frame_bgr)

def save_snapshot(face_bgr, codigo="NA"):
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{codigo}.jpg"
    path = SNAP_DIR / name
    cv2.imwrite(str(path), face_bgr)
    return str(path)

def detectar_camara(max_idx=4):
    print("Buscando cámara activa...")
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        ok, _ = cap.read()
        print(f"Índice {i}: {'OK' if ok else 'No frame'}")
        cap.release()
        if ok: return i
    print("No se detectó cámara disponible.")
    return None

def decidir_identidad(encoding):
    if len(known_encodings) == 0:
        return "Sujeto no identificado", None, 1.0
    distances = face_recognition.face_distance(known_encodings, encoding)
    order = np.argsort(distances)
    best_idx = int(order[0])
    best_dist = float(distances[best_idx])
    second_best = float(distances[order[1]]) if len(order) > 1 else 1.0
    if (best_dist <= THRESH) and ((second_best - best_dist) >= MARGIN):
        return known_names[best_idx], best_dist, second_best
    return "Sujeto no identificado", best_dist, second_best

def loop_panel(cam_id=0, sleep_s=0.6):
    recent_votes = deque(maxlen=VOTES_WINDOW)
    ensure_csv_header()
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("No se pudo abrir la cámara en modo panel; escribiendo frame placeholder.")
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8); frame_bgr[:] = (0, 140, 255)
        save_frame(frame_bgr); time.sleep(1.0)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8); frame_bgr[:] = (0, 140, 255)
            cv2.putText(frame_bgr, "Sin señal de cámara", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 2)
            save_frame(frame_bgr); time.sleep(1.0); continue

        frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_frame, model=DETECTOR_MODEL)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        had_faces = False
        for (encoding, loc) in zip(encodings, locations):
            had_faces = True
            candidate, best_dist, second_best = decidir_identidad(encoding)

            recent_votes.append(candidate)
            final_name, votes = Counter(recent_votes).most_common(1)[0]

            if final_name != "Sujeto no identificado" and votes >= VOTES_NEED:
                shown_name = final_name; color = (0, 255, 0); decision = "accepted"
            else:
                shown_name = "Sujeto no identificado"; color = (0, 0, 255); decision = "rejected"

            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, shown_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            h, w = frame.shape[:2]
            y1, y2 = max(0, top), min(h, bottom)
            x1, x2 = max(0, left), min(w, right)
            face_crop = frame[y1:y2, x1:x2].copy() if y2>y1 and x2>x1 else None
            snap_path = save_snapshot(face_crop, codigo=shown_name) if face_crop is not None else ""

            quality = "high" if len(locations) <= 3 else "mid"
            append_event(str(cam_id),
                         name=shown_name if shown_name!="Sujeto no identificado" else "",
                         codigo=shown_name if shown_name!="Sujeto no identificado" else "",
                         grado="",
                         distancia=f"{best_dist:.4f}" if best_dist is not None else "",
                         decision=decision,
                         quality=quality,
                         snapshot_path=snap_path)

        if not had_faces:
            cv2.putText(frame, "No se detecta ningún rostro", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        save_frame(frame)
        time.sleep(sleep_s)

def loop_ui(cam_id=0):
    recent_votes = deque(maxlen=VOTES_WINDOW)
    ensure_csv_header()
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened(): raise RuntimeError("No se pudo abrir la cámara en modo UI.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame, model=DETECTOR_MODEL)
        encodings = face_recognition.face_encodings(rgb_frame, locations)
        for (encoding, loc) in zip(encodings, locations):
            candidate, best_dist, _ = decidir_identidad(encoding)
            recent_votes.append(candidate)
            final_name, votes = Counter(recent_votes).most_common(1)[0]
            if final_name != "Sujeto no identificado" and votes >= VOTES_NEED:
                shown_name = final_name; color = (0,255,0)
            else:
                shown_name = "Sujeto no identificado"; color = (0,0,255)
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, shown_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            save_frame(frame)
        if len(encodings) == 0:
            cv2.putText(frame, "No se detecta ningún rostro", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            save_frame(frame)
        cv2.imshow("Reconocimiento Facial Universal", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["panel","ui"], default="panel")
    ap.add_argument("--cam", type=int, default=None)
    args = ap.parse_args()

    cam_idx = args.cam if args.cam is not None else detectar_camara()
    if cam_idx is None:
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8); frame_bgr[:] = (0, 140, 255)
        save_frame(frame_bgr); print("Sin cámara; escrito frame placeholder."); time.sleep(2)

    if args.mode == "panel":
        loop_panel(cam_id=cam_idx if cam_idx is not None else 0)
    else:
        loop_ui(cam_id=cam_idx if cam_idx is not None else 0)

if __name__ == "__main__":
    main()
