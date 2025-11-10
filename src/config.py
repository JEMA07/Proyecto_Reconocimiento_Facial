from pathlib import Path

# Raíz del proyecto (este archivo está en src/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Rutas de datos
DATA_DIR   = BASE_DIR / "data"
LOGS_DIR   = DATA_DIR / "logs"
SNAP_DIR   = DATA_DIR / "snapshots"
RUN_DIR    = DATA_DIR / "run"            # <- ESTA ES LA NUEVA CONSTANTE
LAST_FRAME = DATA_DIR / "last_frame.jpg"
EVENTS_CSV = LOGS_DIR / "events.csv"

# Crear carpetas necesarias
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
