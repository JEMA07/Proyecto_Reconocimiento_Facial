import argparse, time, csv, os
from datetime import datetime
from collections import deque, Counter
import cv2
import numpy as np
import face_recognition
import pickle
from pathlib import Path
from scipy.spatial import distance as dist
from src.config import LAST_FRAME, EVENTS_CSV, SNAP_DIR, RUN_DIR
from src.capture_faces import open_any, read_loop
import src.analytics as analytics  # Tu módulo de inteligencia

# --- 1. CARGA DEL MODELO ---
MODEL_PATH = os.path.join("models", "embeddings_mtcnn.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontró el modelo entrenado.")
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"✅ Base cargada con {len(known_names)} rostros registrados.")

# --- 2. CONFIGURACIÓN ---
THRESH = 0.50
MARGIN = 0.07
DETECTOR_MODEL = "hog"
VOTES_WINDOW = 7
VERBOSE = True
SNAPSHOT_COOLDOWN = 20.0
MAX_SNAPSHOTS = 10
FRAME_SKIP = 2
EYE_AR_THRESH = 0.25 # Umbral de parpadeo

def log(msg):
    if VERBOSE: print(msg)

def write_status(text):
    try: (RUN_DIR / "vision.status").write_text(str(text), encoding="utf-8")
    except Exception: pass

def ensure_csv_header():
    if not EVENTS_CSV.exists():
        EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","cam_id","name","codigo","grado","distancia","decision","quality","snapshot_path"])

def append_event(cam_id, name, codigo, grado, distancia, decision, quality, snapshot_path):
    ensure_csv_header()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with EVENTS_CSV.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts, cam_id, name, codigo, grado, distancia, decision, quality, snapshot_path])
    except Exception: pass

# --- CORRECCIÓN CRÍTICA AQUÍ ---
def save_frame_atomic(frame_bgr, path=LAST_FRAME):
    dirp = Path(path).parent
    dirp.mkdir(parents=True, exist_ok=True)
    
    # Separamos nombre y extensión (ej: "last_frame" y ".jpg")
    root, ext = os.path.splitext(str(path))
    if not ext: ext = ".jpg"
    
    # Creamos un nombre temporal VÁLIDO para OpenCV (ej: "last_frame_tmp.jpg")
    temp_path = f"{root}_tmp{ext}"
    
    ok = cv2.imwrite(temp_path, frame_bgr)
    if ok:
        try: os.replace(temp_path, str(path))
        except: pass
# -------------------------------

def save_snapshot(face_bgr, codigo="NA"):
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    safe_cod = "".join([c for c in codigo if c.isalnum() or c in (' ','_')]).strip().replace(' ','_')
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_cod}.jpg"
    path = SNAP_DIR / name
    cv2.imwrite(str(path), face_bgr)
    return str(path)

def _placeholder_frame():
    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_bgr[:] = (0, 140, 255)
    cv2.putText(frame_bgr, "Esperando video...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 2)
    return frame_bgr

def decidir_identidad(encoding):
    if len(known_encodings) == 0:
        return "DESCONOCIDO", None, 1.0
    distances = face_recognition.face_distance(known_encodings, encoding)
    order = np.argsort(distances)
    best_idx = int(order[0])
    best_dist = float(distances[best_idx])
    second_best = float(distances[order[1]]) if len(order) > 1 else 1.0
    
    if (best_dist <= THRESH) and ((second_best - best_dist) >= MARGIN):
        return known_names[best_idx], best_dist, second_best
    return "DESCONOCIDO", best_dist, second_best

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def loop_panel(cam_id=None, url=None, prefer="auto", sleep_s=0.001):
    recent_votes = deque(maxlen=VOTES_WINDOW)
    ensure_csv_header()
    last_snap_time = {} 
    snap_counts = {}
    liveness_states = {} 

    frame_count = 0
    last_draw_info = [] 

    preferred = [cam_id] if cam_id is not None else None
    orig_url = url
    open_url_first = (prefer == "url") or (prefer == "auto" and url)
    open_local = (prefer == "local") or (prefer == "auto" and not url)
    
    cam_sel, cap, be_name = None, None, ""
    if open_url_first and url: cam_sel, cap, be_name = open_any(url=url, prefer_w=640, prefer_h=480, verbose=True)
    if cap is None and open_local: cam_sel, cap, be_name = open_any(url=None, prefer_w=640, prefer_h=480, preferred_indices=preferred, verbose=True)
    
    if cap is None:
        write_status("cam=None backend=None size=0x0")
        while True: save_frame_atomic(_placeholder_frame()); time.sleep(0.9)
        
    if cam_sel is None and url: reopen_args = (orig_url, 640, 480, None, 8, True)
    else: reopen_args = (None, 640, 480, [cam_sel], 8, True)

    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        write_status(f"cam={cam_sel if cam_sel is not None else 'IP'} backend={be_name} size={w}x{h}")
    except: pass

    # --- BUCLE PRINCIPAL ---
    for ok, frame in read_loop(cap, open_any, reopen_args, max_misses=15, delay_s=0.005, verbose=True):
        if not ok:
            write_status("cam=None backend=None size=0x0")
            save_frame_atomic(_placeholder_frame()); time.sleep(1.0); continue
        if frame is None or frame.size == 0: continue

        frame_count += 1
        
        # --- PROCESAMIENTO (1 de cada 3 frames) ---
        if frame_count % (FRAME_SKIP + 1) == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = frame.shape[:2]

            locations = face_recognition.face_locations(rgb_small_frame, model=DETECTOR_MODEL)
            encodings = face_recognition.face_encodings(rgb_small_frame, locations)
            landmarks_list = face_recognition.face_landmarks(rgb_small_frame, locations)

            current_draw_info = [] 

            for (encoding, loc, landmarks) in zip(encodings, locations, landmarks_list):
                # 1. IDENTIDAD
                candidate, best_dist, _ = decidir_identidad(encoding)
                recent_votes.append(candidate)
                final_name, votes = Counter(recent_votes).most_common(1)[0]
                
                # 2. LIVENESS (PARPADEO)
                try:
                    leftEye = landmarks['left_eye']
                    rightEye = landmarks['right_eye']
                    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                    if ear < EYE_AR_THRESH: liveness_states[final_name] = True
                except: pass
                
                is_alive = liveness_states.get(final_name, False)

                # 3. COORDENADAS ORIGINALES
                top, right, bottom, left = loc
                top *= 4; right *= 4; bottom *= 4; left *= 4

                # 4. ANALITICA AVANZADA
                def to_orig(p): return (p[0]*4, p[1]*4)
                shape = {
                    30: to_orig(landmarks['nose_tip'][0]),
                    8:  to_orig(landmarks['chin'][0]),
                    36: to_orig(landmarks['left_eye'][0]),
                    45: to_orig(landmarks['right_eye'][3]),
                    48: to_orig(landmarks['top_lip'][0]),
                    54: to_orig(landmarks['top_lip'][6])
                }
                
                attn_status, attn_color, nose_pt = analytics.get_head_pose(shape, w_orig, h_orig)

                # 5. EMOCIÓN
                face_crop = frame[max(0, top):min(h_orig, bottom), max(0, left):min(w_orig, right)]
                emotion = "-"
                if face_crop.size > 0:
                    emotion = analytics.get_emotion(final_name, face_crop)

                # --- DECISIÓN ---
                if final_name == "DESCONOCIDO":
                    main_color = (0, 165, 255) # Naranja
                    label_top = f"ALERTA: {final_name}"
                    label_bot = "NO AUTORIZADO"
                    decision = "ALERTA"
                elif not is_alive:
                    main_color = (0, 255, 255) # Amarillo
                    label_top = f"{final_name} ({emotion})"
                    label_bot = "PARPADEE POR FAVOR"
                    decision = "LIVENESS"
                else:
                    main_color = attn_color 
                    label_top = f"{final_name} | {emotion}"
                    label_bot = f"ACCESO | {attn_status}"
                    decision = "ACCESO"

                current_draw_info.append({
                    "rect": (left, top, right, bottom),
                    "color": main_color,
                    "top_text": label_top,
                    "bot_text": label_bot,
                    "nose": nose_pt,
                    "status": attn_status
                })

                # SNAPSHOTS
                snap_path = ""
                current_count = snap_counts.get(final_name, 0)
                if current_count < MAX_SNAPSHOTS and (decision in ["ALERTA", "ACCESO"]):
                    now = time.time()
                    if (now - last_snap_time.get(final_name, 0)) > SNAPSHOT_COOLDOWN:
                        try:
                            snap_path = save_snapshot(face_crop, codigo=final_name)
                            last_snap_time[final_name] = now 
                            snap_counts[final_name] = current_count + 1
                        except: pass
                
                # CSV
                if decision in ["ACCESO", "ALERTA"]:
                    extra_data = f"{attn_status}|{emotion}"
                    decision_csv = "accepted" if decision == "ACCESO" else "rejected"
                    append_event(str(cam_sel), final_name, final_name, "", f"{best_dist:.2f}", decision_csv, extra_data, snap_path)

            last_draw_info = current_draw_info

        # --- DIBUJAR ---
        for info in last_draw_info:
            l, t, r, b = info["rect"]
            col = info["color"]
            
            cv2.rectangle(frame, (l, t), (r, b), col, 2)
            
            # Etiquetas
            cv2.rectangle(frame, (l, t - 30), (r, t), col, cv2.FILLED)
            cv2.putText(frame, info["top_text"], (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            cv2.rectangle(frame, (l, b), (r, b + 25), (0,0,0), cv2.FILLED)
            cv2.putText(frame, info["bot_text"], (l + 5, b + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            nose = info["nose"]
            cv2.circle(frame, nose, 5, col, -1)

        save_frame_atomic(frame)
        time.sleep(sleep_s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["panel","ui"], default="panel")
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--url", type=str, default=os.getenv("CAM_URL", "").strip())
    ap.add_argument("--prefer", choices=["auto","url","local"], default="auto")
    args = ap.parse_args()
    if args.url: os.environ["CAM_URL"] = args.url
    loop_panel(cam_id=args.cam, url=args.url, prefer=args.prefer)

if __name__ == "__main__":
    main()