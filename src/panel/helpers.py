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
        # MEJORA 1: on_bad_lines='skip' evita crasheos si hay líneas corruptas
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines='skip')
        
        # Asegurar que existan todas las columnas
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
                
        # Limpieza de tipos
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["distancia"] = pd.to_numeric(df["distancia"], errors="coerce")
        
        # MEJORA 2: Ordenar por fecha (ascendente) para consistencia
        df = df.sort_values("timestamp", ascending=True)
        
        # Limitar tamaño para que el panel no se ponga lento con el tiempo
        if len(df) > max_rows:
            df = df.tail(max_rows).copy()
            
        return df
    except Exception as e:
        # Si algo falla drásticamente, devolvemos vacío pero NO rompemos el panel
        print(f"Error leyendo CSV: {e}")
        return pd.DataFrame(columns=COLUMNS)

def eventos_hoy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns: return df
    try:
        hoy = date.today()
        # Filtramos solo los que tienen fecha válida
        mask = df["timestamp"].dt.date == hoy
        return df[mask].copy()
    except:
        return df

def ultimo_evento(df: pd.DataFrame) -> Optional[Dict]:
    if df.empty: return None
    try:
        # Retornamos el último de la lista (el más reciente)
        return df.iloc[-1].to_dict()
    except:
        return None

def metricas(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {"total": 0, "identificados": 0, "porc_ident": 0.0, "ultimo_ts": None}
        
    total = int(len(df))
    # Contamos cuántos fueron "accepted"
    ident = int((df["decision"] == "accepted").sum())
    pct = round((ident / total)*100, 1) if total > 0 else 0.0
    last_ts = df.iloc[-1]["timestamp"]
    
    return {"total": total, "identificados": ident, "porc_ident": pct, "ultimo_ts": last_ts}

def recientes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty: return df
    # Devolvemos los últimos N, invertidos (el más nuevo arriba)
    return df.tail(n).iloc[::-1].copy()

def cargar_frame(frame_path: Path):
    """Carga imagen usando PIL (Útil para reportes estáticos, no para video en vivo)"""
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