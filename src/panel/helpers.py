from __future__ import annotations
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional
import pandas as pd
from PIL import Image
import io

COLUMNS = ["timestamp","cam_id","name","codigo","grado","distancia","decision","quality","snapshot_path"]

def leer_eventos(csv_path: Path, max_rows: int = 5000) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["distancia"] = pd.to_numeric(df["distancia"], errors="coerce")
        if len(df) > max_rows:
            df = df.tail(max_rows).copy()
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

def eventos_hoy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns: return df
    hoy = date.today()
    return df[df["timestamp"].dt.date == hoy].copy()

def ultimo_evento(df: pd.DataFrame) -> Optional[Dict]:
    if df.empty: return None
    return df.iloc[-1].to_dict()

def metricas(df: pd.DataFrame) -> Dict:
    total = int(len(df))
    ident = int((df["decision"] == "accepted").sum()) if not df.empty else 0
    pct = round((ident / total)*100, 1) if total > 0 else 0.0
    last_ts = df.iloc[-1]["timestamp"] if total > 0 else None
    return {"total": total, "identificados": ident, "porc_ident": pct, "ultimo_ts": last_ts}

def recientes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty: return df
    return df.tail(n).iloc[::-1].copy()

def cargar_frame(frame_path: Path):
    if not frame_path.exists(): return None
    try:
        with frame_path.open("rb") as f:
            data = f.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception:
        return None

def exportar_hoy(df: pd.DataFrame, exports_dir: Path):
    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = exports_dir / f"events_{stamp}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    return out

def calidad_color(quality: str) -> str:
    q = (quality or "").lower()
    if q == "high": return "✅"
    if q == "mid":  return "🟡"
    if q == "low":  return "🔴"
    return "⚪"
