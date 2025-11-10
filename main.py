# Punto de entrada principal del sistema
import argparse
import subprocess
import sys
from pathlib import Path

def run_panel(port: int = 8501):
    app_path = Path("src/panel/app_streamlit.py").resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--server.address", "0.0.0.0"]
    return subprocess.call(cmd)

def run_vision():
    # Ejecuta tu proceso de reconocimiento como módulo
    return subprocess.call([sys.executable, "-m", "src.recognize"])

def main():
    p = argparse.ArgumentParser(description="Punto de entrada del sistema.")
    p.add_argument("--mode", choices=["vision","panel"], default="panel")
    p.add_argument("--port", type=int, default=8501)
    args = p.parse_args()

    if args.mode == "panel":
        sys.exit(run_panel(port=args.port))
    else:
        sys.exit(run_vision())

if __name__ == "__main__":
    main()
