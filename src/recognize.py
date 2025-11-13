import argparse, time, csv, os, uuid
from datetime import datetime
from collections import deque, Counter
import cv2
import numpy as np
import face_recognition
import pickle
from pathlib import Path
from src.config import LAST_FRAME, EVENTS_CSV, SNAP_DIR, RUN_DIR
from src.capture_faces import open_any, read_loop  # capturador universal

# --- CARGA DEL MODELO ---
MODEL_PATH = os.path.join("models", "embeddings_mtcnn.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontró el modelo entrenado.")
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]; known_names = data["names"]
print(f"Base cargada con {len(known_names)} rostros registrados.")

# --- CONFIGURACIÓN ---
THRESH = 0.50; MARGIN = 0.07
DETECTOR_MODEL = "hog"
VOTES_WINDOW = 7; VOTES_NEED = 5
VERBOSE = True
SNAPSHOT_COOLDOWN = 5.0  # <--- NUEVO: Segundos de espera entre fotos de la misma persona

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
    with EVENTS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, cam_id, name, codigo, grado, distancia, decision, quality, snapshot_path])

def save_frame_atomic(frame_bgr, path=LAST_FRAME):
    dirp = Path(path).parent
    dirp.mkdir(parents=True, exist_ok=True)
    root, ext = os.path.splitext(str(LAST_FRAME)); ext = ext or ".jpg"
    last = os.path.join(str(dirp), f"last_frame{ext}")
    prev = os.path.join(str(dirp), f"last_frame_prev{ext}")
    nextp = os.path.join(str(dirp), f"last_frame_next{ext}")

    ok = cv2.imwrite(nextp, frame_bgr)
    if not ok: return # Evitamos crash si falla escritura
    
    time.sleep(0.01)
    try:
        if os.path.exists(last):
            try:
                if os.path.exists(prev):
                    try: os.remove(prev)
                    except: pass
                os.replace(last, prev)
            except PermissionError: pass
    except Exception: pass

    for _ in range(6):
        try:
            os.replace(nextp, last)
            return
        except PermissionError:
            time.sleep(0.03)
        except Exception: break
    
    # Fallback copy
    try:
        import shutil
        shutil.copyfile(nextp, last)
        try: os.remove(nextp)
        except: pass
    except Exception: pass

def save_snapshot(face_bgr, codigo="NA"):
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    # Nombre limpio sin caracteres especiales
    safe_cod = "".join([c for c in codigo if c.isalnum() or c in (' ','_')]).strip().replace(' ','_')
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_cod}.jpg"
    path = SNAP_DIR / name
    cv2.imwrite(str(path), face_bgr)
    return str(path)

def _placeholder_frame():
    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_bgr[:] = (0, 140, 255)
    cv2.putText(frame_bgr, "Sin senal de camara", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 2)
    return frame_bgr

def decidir_identidad(encoding):
    if len(known_encodings) == 0:
        return "Sujeto no identificado", None, 1.0
    distances = face_recognition.face_distance(known_encodings, encoding)
    order = np.argsort(distances)
    best_idx = int(order[0]); best_dist = float(distances[best_idx])
    second_best = float(distances[order[1]]) if len(order) > 1 else 1.0
    if (best_dist <= THRESH) and ((second_best - best_dist) >= MARGIN):
        return known_names[best_idx], best_dist, second_best
    return "Sujeto no identificado", best_dist, second_best

def loop_panel(cam_id=None, url=None, prefer="auto", sleep_s=0.7):
    recent_votes = deque(maxlen=VOTES_WINDOW)
    ensure_csv_header()

    # --- FRENO DE FOTOS: Diccionario para guardar tiempos ---
    last_snap_time = {} 
    # --------------------------------------------------------

    preferred = [cam_id] if cam_id is not None else None
    orig_url = url

    open_url_first = (prefer == "url") or (prefer == "auto" and url)
    open_local = (prefer == "local") or (prefer == "auto" and not url)

    cam_sel, cap, be_name = None, None, ""
    
    # Intento 1: URL
    if open_url_first and url:
        cam_sel, cap, be_name = open_any(url=url, prefer_w=640, prefer_h=480, preferred_indices=None, verbose=True)
        if cap is None and prefer == "url":
            write_status("cam=None backend=None size=0x0")
            while True: save_frame_atomic(_placeholder_frame()); time.sleep(0.9)

    # Intento 2: Local
    if cap is None and open_local:
        cam_sel, cap, be_name = open_any(url=None, prefer_w=640, prefer_h=480, preferred_indices=preferred, verbose=True)

    # Fallo total
    if cap is None:
        write_status("cam=None backend=None size=0x0")
        while True: save_frame_atomic(_placeholder_frame()); time.sleep(0.9)

    # Configurar reapertura
    if cam_sel is None and url:
        reopen_args = (orig_url, 640, 480, None, 8, True)
    else:
        reopen_args = (None, 640, 480, [cam_sel], 8, True)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    write_status(f"cam={cam_sel if cam_sel is not None else 'IP'} backend={be_name} size={w}x{h}")

    # --- BUCLE PRINCIPAL ---
    for ok, frame in read_loop(cap, open_any, reopen_args, max_misses=15, delay_s=0.02, verbose=True):
        if not ok:
            write_status("cam=None backend=None size=0x0")
            save_frame_atomic(_placeholder_frame())
            # Logica simple de reconexion...
            time.sleep(1.0); continue

        if frame is None or frame.size == 0:
            save_frame_atomic(_placeholder_frame()); time.sleep(0.2); continue

        # --- CORRECCIÓN ROTACIÓN (Celular en vertical) ---
        # Si tu imagen se ve acostada, descomenta la siguiente linea:
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
        # -------------------------------------------------

        frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_frame, model=DETECTOR_MODEL)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        had_faces = False
        for (encoding, loc) in zip(encodings, locations):
            had_faces = True
            candidate, best_dist, _ = decidir_identidad(encoding)

            recent_votes.append(candidate)
            final_name, votes = Counter(recent_votes).most_common(1)[0]
            
            if final_name != "Sujeto no identificado" and votes >= VOTES_NEED:
                shown_name = final_name; color = (0, 255, 0); decision = "accepted"
            else:
                shown_name = "Sujeto no identificado"; color = (0, 0, 255); decision = "rejected"

            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, shown_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # --- LOGICA DE GUARDADO CON FRENO ---
            snap_path = ""
            h_f, w_f = frame.shape[:2]
            y1, y2 = max(0, top), min(h_f, bottom)
            x1, x2 = max(0, left), min(w_f, right)
            
            face_crop = None
            if y2 > y1 and x2 > x1:
                face_crop = frame[y1:y2, x1:x2].copy()
            
            if face_crop is not None:
                now = time.time()
                # Solo guardamos si pasaron 5 segundos desde la ultima foto de ESTA persona
                last_time = last_snap_time.get(shown_name, 0)
                
                if (now - last_time) > SNAPSHOT_COOLDOWN:
                    try:
                        snap_path = save_snapshot(face_crop, codigo=shown_name)
                        last_snap_time[shown_name] = now  # Actualizamos reloj para esta persona
                    except Exception as e:
                        log(f"Error guardando snap: {e}")
            # --------------------------------------

            quality = "high" if len(locations) <= 3 else "mid"
            append_event(str(cam_sel if cam_sel is not None else 0),
                         name=shown_name if shown_name!="Sujeto no identificado" else "",
                         codigo=shown_name if shown_name!="Sujeto no identificado" else "",
                         grado="",
                         distancia=f"{best_dist:.4f}" if best_dist is not None else "",
                         decision=decision,
                         quality=quality,
                         snapshot_path=snap_path)

        if not had_faces:
            cv2.putText(frame, "No se detecta ningun rostro", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        save_frame_atomic(frame)
        try:
            w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            write_status(f"cam={cam_sel if cam_sel is not None else 'IP'} backend={be_name} size={w_cap}x{h_cap}")
        except Exception: pass

        time.sleep(sleep_s)

# --- MODO UI (Ventana Local) ---
def loop_ui(cam_id=None, url=None, prefer="auto"):
    ensure_csv_header()
    last_snap_time = {} # Freno tambien para modo UI
    
    preferred = [cam_id] if cam_id is not None else None
    cam_sel, cap, be_name = None, None, ""
    if (prefer in ("url","auto")) and url:
        cam_sel, cap, be_name = open_any(url=url, prefer_w=640, prefer_h=480, preferred_indices=None, verbose=True)
    if cap is None and (prefer in ("local","auto")):
        cam_sel, cap, be_name = open_any(url=None, prefer_w=640, prefer_h=480, preferred_indices=preferred, verbose=True)
    if cap is None:
        raise RuntimeError("No se pudo abrir ninguna fuente en modo UI.")
    
    recent_votes = deque(maxlen=VOTES_WINDOW)
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # Descomentar si es celular vertical
        
        frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame, model=DETECTOR_MODEL)
        encodings = face_recognition.face_encodings(rgb_frame, locations)
        
        for (encoding, loc) in zip(encodings, locations):
            candidate, best_dist, _ = decidir_identidad(encoding)
            recent_votes.append(candidate)
            final_name, votes = Counter(recent_votes).most_common(1)[0]
            color = (0,255,0) if (final_name!="Sujeto no identificado" and votes>=VOTES_NEED) else (0,0,255)
            
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, final_name if color==(0,255,0) else "Sujeto no identificado", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Freno simple para UI
            now = time.time()
            if (now - last_snap_time.get(final_name, 0)) > SNAPSHOT_COOLDOWN:
                 # Aqui podrias guardar si quisieras
                 last_snap_time[final_name] = now

            save_frame_atomic(frame)
            
        if len(encodings) == 0:
            cv2.putText(frame, "No se detecta ningun rostro", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            save_frame_atomic(frame)
            
        cv2.imshow("Reconocimiento Facial Universal", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["panel","ui"], default="panel")
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--url", type=str, default=os.getenv("CAM_URL", "").strip())
    ap.add_argument("--prefer", choices=["auto","url","local"], default="auto")
    args = ap.parse_args()

    if args.url:
        os.environ["CAM_URL"] = args.url

    if args.mode == "panel":
        loop_panel(cam_id=args.cam, url=args.url, prefer=args.prefer)
    else:
        loop_ui(cam_id=args.cam, url=args.url, prefer=args.prefer)

if __name__ == "__main__":
    main()