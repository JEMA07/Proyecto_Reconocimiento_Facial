from __future__ import annotations
# Asegurar import de 'src' cuando se ejecuta con streamlit run
from pathlib import Path
import sys, time
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.config import BASE_DIR, LOGS_DIR, SNAP_DIR, RUN_DIR, LAST_FRAME, EVENTS_CSV
from src.panel.assets import APP_TITLE, APP_SUBTITLE, REFRESH_MS_DEFAULT, LOGO, DECISION_MARK
from src.panel.helpers import (
    leer_eventos, eventos_hoy, ultimo_evento, metricas, recientes,
    cargar_frame, exportar_hoy, calidad_color
)
from src.panel.control import start_worker, stop_worker, get_pid

PIDFILE = RUN_DIR / "vision.pid"

# ---------------- Config de página ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧠")

# ---------------- Sidebar ----------------
if LOGO.exists():
    st.sidebar.image(str(LOGO), width=140)
st.sidebar.title("Controles")

# PIN de operador
pin_ok = st.session_state.get("pin_ok", False)
if not pin_ok:
    pin = st.sidebar.text_input("PIN de operador", type="password", placeholder="****")
    if st.sidebar.button("Validar PIN"):
        if pin == "1234":  # cambia este PIN
            st.session_state["pin_ok"] = True
            pin_ok = True
            st.sidebar.success("PIN válido")
        else:
            st.sidebar.error("PIN incorrecto")
else:
    st.sidebar.caption("Operador autenticado")

ref_ms = st.sidebar.slider("Refresco (ms)", 500, 5000, REFRESH_MS_DEFAULT, 100)
modo_silencioso = st.sidebar.toggle("Modo silencioso", value=True)
st.sidebar.caption("La voz/alarma se controla desde el proceso de visión.")

# Botonera de control del reconocimiento
st.sidebar.subheader("Servicio de reconocimiento")
pid_actual = get_pid(PIDFILE)
col_a, col_b = st.sidebar.columns(2)
if col_a.button("Iniciar", disabled=not pin_ok or pid_actual is not None):
    pid = start_worker(PIDFILE, module="src.recognize")
    if pid:
        st.sidebar.success(f"Iniciado (PID {pid})")
    else:
        st.sidebar.error("No se pudo iniciar. Revisa logs del worker.")

if col_b.button("Detener", disabled=not pin_ok or pid_actual is None):
    if stop_worker(PIDFILE):
        st.sidebar.warning("Servicio detenido")
    else:
        st.sidebar.error("No se pudo detener. Verifica permisos.")

# Estado
pid_actual = get_pid(PIDFILE)
if pid_actual:
    st.sidebar.info(f"Estado: RUNNING (PID {pid_actual})")
else:
    st.sidebar.warning("Estado: STOPPED")

# ---------------- Encabezado ----------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# ---------------- Layout principal ----------------
top_kpis = st.container()
col_izq, col_der = st.columns([2.2, 1.4], gap="large")

def render_once():
    df = leer_eventos(EVENTS_CSV)
    df_hoy = eventos_hoy(df)
    m = metricas(df_hoy)

    # KPIs
    with top_kpis:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Eventos hoy", m["total"])
        k2.metric("% identificados", f"{m['porc_ident']}%")
        k3.metric("Identificados", m["identificados"])
        k4.metric("Último evento", m["ultimo_ts"].strftime("%H:%M:%S") if m["ultimo_ts"] else "—")

    # En vivo
    with col_izq:
        st.subheader("En vivo")
        img = cargar_frame(LAST_FRAME)
        if img is not None:
            st.image(img, channels="RGB", use_column_width=True)
        else:
            if pid_actual:
                st.info("Sin frame aún... El worker está iniciando o no ha recibido video.")
            else:
                st.info("Servicio detenido. Inicia el reconocimiento para ver el video.")
        st.caption(f"Frame: {LAST_FRAME} | CSV: {EVENTS_CSV}")

    # Ficha + recientes
    with col_der:
        st.subheader("Ficha actual")
        evt = ultimo_evento(df_hoy)
        if evt:
            nombre = evt.get("name", "—")
            codigo = evt.get("codigo", "—")
            grado  = evt.get("grado", "—")
            dist   = evt.get("distancia", "—")
            dec    = str(evt.get("decision", "—"))
            qual   = str(evt.get("quality", "—"))
            hora   = evt.get("timestamp").strftime("%H:%M:%S") if evt.get("timestamp") else "—"
            marca  = DECISION_MARK.get(dec, "⚪")
            st.markdown(f"**{marca} {nombre}**")
            st.write(f"Código: {codigo} | Grado: {grado}")
            st.write(f"Distancia: {dist} | Calidad: {calidad_color(qual)} | Hora: {hora}")
            snap = evt.get("snapshot_path")
            if snap and Path(snap).exists():
                st.image(str(snap), caption="Snapshot", width=260)
        else:
            st.info("Sin eventos de hoy aún.")

        st.divider()
        st.subheader("Eventos recientes")
        df_r = recientes(df_hoy, n=10)
        if df_r.empty:
            st.write("No hay eventos recientes.")
        else:
            mostrar = df_r[["timestamp","name","codigo","grado","distancia","decision","quality"]].copy()
            mostrar["timestamp"] = mostrar["timestamp"].dt.strftime("%H:%M:%S")
            st.dataframe(mostrar, use_container_width=True, hide_index=True)

        st.divider()
        if st.button("Exportar CSV de hoy"):
            out = exportar_hoy(df_hoy, LOGS_DIR / "exports")
            st.success(f"Exportado: {out}")

render_once()

# Autorefresco ligero
time.sleep(ref_ms / 1000.0)
