# src/capture.py
# Capturador universal: URL/RTSP o cámara local con DSHOW/MSMF y warm‑up.
import os, time, platform
from typing import Optional, Tuple, List
import cv2

def _log(msg: str, verbose=True):
    if verbose:
        print(msg)

def _warmup(cap: cv2.VideoCapture, tries=12, delay=0.02) -> Tuple[bool, Optional[any]]:
    """Lee varios frames para estabilizar; algunos backends devuelven el primero vacío."""
    ok = False; frame = None
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(delay)
    return False, None

def list_local_cameras(max_idx=10, verbose=False) -> List[int]:
    """Devuelve índices que parecen abrirse (no garantiza frames, solo apertura)."""
    found = []
    for idx in range(max_idx):
        cap = cv2.VideoCapture(idx)  # backend por defecto
        if cap.isOpened():
            found.append(idx)
        cap.release()
    _log(f"[ENUM] Cámaras locales detectadas: {found}", verbose)
    return found

def open_local_camera(idx: int, prefer_w=640, prefer_h=480, verbose=True) -> Tuple[Optional[cv2.VideoCapture], str]:
    """
    Abre una cámara local por índice de forma robusta.
    Windows: intenta DirectShow y luego Media Foundation. Otros: backend por defecto.
    """
    sysname = platform.system().lower()
    if "windows" in sysname:
        backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
    else:
        backends = [(cv2.CAP_ANY, "ANY")]

    for be, name in backends:
        _log(f"[TRY] idx={idx} backend={name} {prefer_w}x{prefer_h}", verbose)
        cap = cv2.VideoCapture(idx, be)
        if not cap.isOpened():
            _log(f"[FAIL-OPEN] idx={idx} backend={name}", verbose)
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  prefer_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, prefer_h)

        ok, frame = _warmup(cap)
        if not ok:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = _warmup(cap)

        if ok and frame is not None and frame.size > 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _log(f"[OK] Cam {idx} backend {name} ({w}x{h})", verbose)
            return cap, name

        _log(f"[FAIL-FRAME] idx={idx} backend={name}", verbose)
        cap.release()

    return None, ""

def open_url(source_url: str, verbose=True) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Abre una fuente de red (HTTP/RTSP/archivo) vía backend de archivos/FFmpeg de OpenCV."""
    _log(f"[TRY] URL {source_url}", verbose)
    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        _log(f"[FAIL-OPEN] URL", verbose)
        cap.release()
        return None, ""
    ok, frame = _warmup(cap)
    if ok and frame is not None and frame.size > 0:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _log(f"[OK] URL ({w}x{h})", verbose)
        return cap, "URL"
    _log("[FAIL-FRAME] URL", verbose)
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
    Selecciona y abre una fuente de video. Devuelve (cam_id, cap, backend_name).
    Nota: si 'url' se proporciona, NO se escanean cámaras locales desde aquí.
    """
    # 1) URL explícita (decisión de caer a locales se hace en nivel superior)
    if url:
        cap, be = open_url(url, verbose=verbose)
        return None, cap, be  # cap puede ser None si falla

    # 2) Índices preferidos
    if preferred_indices:
        for idx in preferred_indices:
            cap, be = open_local_camera(idx, prefer_w, prefer_h, verbose=verbose)
            if cap is not None:
                return idx, cap, be

    # 3) Escaneo de índices
    for idx in list_local_cameras(max_idx=scan_limit, verbose=verbose):
        cap, be = open_local_camera(idx, prefer_w, prefer_h, verbose=verbose)
        if cap is not None:
            return idx, cap, be

    _log("[ERROR] No se pudo abrir ninguna fuente de video", verbose)
    return None, None, ""

def read_loop(cap: cv2.VideoCapture, reopen_fn, reopen_args: tuple, max_misses=15, delay_s=0.02, verbose=True):
    """
    Generador de frames con autorecuperación: reabre la MISMA fuente al perder demasiadas lecturas.
    'reopen_args' debe ser lo bastante específico para reabrir exactamente la misma fuente.
    """
    misses = 0
    while True:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            misses = 0
            yield True, frame
            continue
        misses += 1
        time.sleep(delay_s)
        if misses >= max_misses:
            _log("[INFO] Reabriendo fuente por lecturas fallidas...", verbose)
            try:
                cap.release()
            except Exception:
                pass
            _, cap, _ = reopen_fn(*reopen_args)
            if cap is None:
                yield False, None
                break
            misses = 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=os.getenv("CAM_URL", "").strip(), help="Fuente de red (http/rtsp/archivo)")
    parser.add_argument("--cam", type=int, default=None, help="Índice de cámara local preferido")
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
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release(); cv2.destroyAllWindows()
