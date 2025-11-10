from __future__ import annotations
from pathlib import Path
import subprocess, sys, os, time, signal
from typing import Optional

def _write_pid(pidfile: Path, pid: int):
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(pid), encoding="utf-8")

def _read_pid(pidfile: Path) -> Optional[int]:
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def _pid_alive(pid: int) -> bool:
    if pid is None:
        return False
    try:
        # Windows: os.kill con 0 no existe; usamos psutil si hiciera falta.
        # Truco portátil: en Windows, abrir el proceso puede requerir psutil; aquí probamos señal.
        if os.name == "nt":
            # En Windows no hay señales POSIX; probamos con 'tasklist'
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            return str(pid) in out.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False

def get_pid(pidfile: Path) -> Optional[int]:
    pid = _read_pid(pidfile)
    if pid and _pid_alive(pid):
        return pid
    # Limpieza de pidfile huérfano
    if pidfile.exists():
        pidfile.unlink(missing_ok=True)
    return None

def start_worker(pidfile: Path, module: str = "src.recognize") -> Optional[int]:
    existing = get_pid(pidfile)
    if existing:
        return existing
    # Lanzar reconocimiento como subproceso
    proc = subprocess.Popen([sys.executable, "-m", module],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=(subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0))
    _write_pid(pidfile, proc.pid)
    # Breve espera y verificación
    time.sleep(0.8)
    if not _pid_alive(proc.pid):
        # Falló al iniciar
        if pidfile.exists():
            pidfile.unlink(missing_ok=True)
        return None
    return proc.pid

def stop_worker(pidfile: Path, timeout: float = 5.0) -> bool:
    pid = get_pid(pidfile)
    if not pid:
        return True
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not _pid_alive(pid):
                break
            time.sleep(0.2)
        if _pid_alive(pid):
            # Forzar kill
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGKILL)
        # Limpieza
        if pidfile.exists():
            pidfile.unlink(missing_ok=True)
        return True
    except Exception:
        return False
