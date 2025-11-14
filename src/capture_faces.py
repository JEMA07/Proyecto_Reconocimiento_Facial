# src/capture_faces.py
# Capturador universal: URL/RTSP o cámara local.
# Actualizado por mí para que DroidCam funcione fluido sin pantallas azules.

import os, time, platform
from typing import Optional, Tuple, List
import cv2

def _log(msg: str, verbose=True):
    if verbose:
        print(msg)

def _warmup(cap: cv2.VideoCapture, tries=12, delay=0.02) -> Tuple[bool, Optional[any]]:
    """
    Le doy un tiempo a la cámara para que 'caliente'. 
    A veces los primeros frames salen negros o vacíos, así que leo varios antes de empezar.
    """
    ok = False; frame = None
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(delay)
    return False, None

def list_local_cameras(max_idx=10, verbose=False) -> List[int]:
    """
    Hago un barrido rápido para ver qué índices de cámara (0, 1, 2...) responden.
    """
    found = []
    for idx in range(max_idx):
        cap = cv2.VideoCapture(idx) 
        if cap.isOpened():
            found.append(idx)
        cap.release()
    _log(f"[ENUM] Cámaras locales que detecté: {found}", verbose)
    return found

def open_local_camera(idx: int, prefer_w=640, prefer_h=480, verbose=True) -> Tuple[Optional[cv2.VideoCapture], str]:
    """
    Aquí es donde abro la cámara local.
    NOTA IMPORTANTE: Cambié la lógica para usar 'AUTO' en lugar de forzar 'DSHOW'.
    Esto arregla el problema de la pantalla azul/sólida con DroidCam en Windows.
    """
    
    # --- MI CORRECCIÓN ESTRATÉGICA ---
    # Antes tenía una lista con cv2.CAP_DSHOW para Windows, pero eso me daba problemas.
    # Ahora uso cv2.CAP_ANY para dejar que el sistema elija el mejor driver automáticamente.
    backends = [(cv2.CAP_ANY, "AUTO")] 
    # ---------------------------------

    for be, name in backends:
        _log(f"[TRY] Intentando abrir cámara {idx} con modo {name} a {prefer_w}x{prefer_h}", verbose)
        cap = cv2.VideoCapture(idx, be)
        
        if not cap.isOpened():
            _log(f"[FAIL-OPEN] No pude abrir la cámara {idx} con {name}", verbose)
            cap.release()
            continue

        # Configuro la resolución que quiero para mi proyecto
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  prefer_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, prefer_h)

        # Hago el calentamiento para asegurar que la imagen venga bien
        ok, frame = _warmup(cap)
        if not ok:
            # Si falla, intento forzar una resolución alta por si es una cámara HD caprichosa
            _log("[WARN] Falló warmup a baja resolución, probando HD...", verbose)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = _warmup(cap)

        if ok and frame is not None and frame.size > 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _log(f"[OK] ¡Éxito! Cámara {idx} abierta con {name} ({w}x{h})", verbose)
            return cap, name

        _log(f"[FAIL-FRAME] La cámara abrió pero no me dio imagen (pantalla negra/vacía)", verbose)
        cap.release()

    return None, ""

def open_url(source_url: str, verbose=True) -> Tuple[Optional[cv2.VideoCapture], str]:
    """
    Para abrir cámaras IP o archivos de video cuando uso la estrategia 'URL'.
    """
    _log(f"[TRY] Intentando conectar a URL: {source_url}", verbose)
    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        _log(f"[FAIL-OPEN] No pude conectar a la URL", verbose)
        cap.release()
        return None, ""
    
    ok, frame = _warmup(cap)
    if ok and frame is not None and frame.size > 0:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _log(f"[OK] Conectado a URL ({w}x{h})", verbose)
        return cap, "URL"
    
    _log("[FAIL-FRAME] Conecté a la URL pero no recibo video", verbose)
    cap.release()
    return None, ""

def open_any(
    url: Optional[str] = None,
    prefer_w=640,
    prefer_h=480,
    preferred_indices: Optional[List[int]] = None,
    scan_limit=8,
    verbose=True
) -> Tuple[Optional[int], Optional[cv2.VideoCapture], str]:
    """
    Mi función maestra para decidir qué abrir.
    Prioridad: 1. URL (si la puse) -> 2. Mis cámaras favoritas -> 3. Escaneo general.
    """
    # 1) Si definí una URL, voy directo a ella.
    if url:
        cap, be = open_url(url, verbose=verbose)
        return None, cap, be 

    # 2) Si tengo índices preferidos (ej: la cámara 1 o 0), pruebo esos primero.
    if preferred_indices:
        for idx in preferred_indices:
            cap, be = open_local_camera(idx, prefer_w, prefer_h, verbose=verbose)
            if cap is not None:
                return idx, cap, be

    # 3) Si nada funcionó, busco cualquier cámara disponible (Escaneo).
    _log("[INFO] Escaneando otras cámaras...", verbose)
    for idx in list_local_cameras(max_idx=scan_limit, verbose=verbose):
        cap, be = open_local_camera(idx, prefer_w, prefer_h, verbose=verbose)
        if cap is not None:
            return idx, cap, be

    _log("[ERROR] Lo intenté todo pero no encontré ninguna fuente de video.", verbose)
    return None, None, ""

def read_loop(cap: cv2.VideoCapture, reopen_fn, reopen_args: tuple, max_misses=15, delay_s=0.02, verbose=True):
    """
    Generador de frames resiliente. 
    Si la cámara se desconecta (ej: fallo de WiFi), intento reconectarla automáticamente
    para que el sistema no se caiga.
    """
    misses = 0
    while True:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            misses = 0
            yield True, frame
            continue
        
        # Si falla la lectura, cuento los errores.
        misses += 1
        time.sleep(delay_s)
        
        # Si falló muchas veces seguidas, asumo que se cayó la conexión y trato de revivirla.
        if misses >= max_misses:
            _log("[INFO] Perdí la señal... intentando reconectar la fuente...", verbose)
            try:
                cap.release()
            except Exception:
                pass
            
            # Llamo a la función de reapertura con los mismos argumentos originales
            _, cap, _ = reopen_fn(*reopen_args)
            if cap is None:
                _log("[FATAL] No pude reconectar. Me rindo.", verbose)
                yield False, None
                break
            
            _log("[INFO] ¡Reconexión exitosa! Seguimos.", verbose)
            misses = 0

if __name__ == "__main__":
    # Bloque de pruebas para correr este archivo solo
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=os.getenv("CAM_URL", "").strip(), help="Fuente de red")
    parser.add_argument("--cam", type=int, default=None, help="Índice de cámara local")
    parser.add_argument("--w", type=int, default=640); parser.add_argument("--h", type=int, default=480)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    preferred = [args.cam] if args.cam is not None else None
    cam_id, cap, backend = open_any(url=args.url, prefer_w=args.w, prefer_h=args.h, preferred_indices=preferred, verbose=args.verbose)
    if cap is None:
        raise SystemExit("No se pudo abrir ninguna fuente de video.")

    print(f"Fuente abierta: cam_id={cam_id} backend={backend}")
    for ok, frame in read_loop(cap, open_any, (args.url, args.w, args.h, preferred, 8, args.verbose), verbose=args.verbose):
        if not ok:
            print("Fuente perdida y no pudo reabrirse.")
            break
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC para salir
            break
    cap.release(); cv2.destroyAllWindows()