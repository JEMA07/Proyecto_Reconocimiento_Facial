from pathlib import Path

# Raíz del proyecto (este archivo está en src/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Rutas de datos
DATA_DIR   = BASE_DIR / "data"
LOGS_DIR   = DATA_DIR / "logs"
SNAP_DIR   = DATA_DIR / "snapshots"
LAST_FRAME = DATA_DIR / "last_frame.jpg"
EVENTS_CSV = LOGS_DIR / "events.csv"

# Asegura carpetas
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)
