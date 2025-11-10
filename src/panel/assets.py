from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

APP_TITLE = "Panel de Reconocimiento Facial"
APP_SUBTITLE = "Video en vivo y eventos recientes"
REFRESH_MS_DEFAULT = 1500  # 1.5 s

LOGO = ASSETS_DIR / "logo.png"  # opcional

DECISION_MARK = {
    "accepted": "🟢",
    "rejected": "🔴",
    "discarded": "⚪",
}
