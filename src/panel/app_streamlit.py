import os, sys, time, pathlib, datetime, socket
import streamlit as st
import pandas as pd

from src.panel.assets import APP_TITLE, APP_SUBTITLE, REFRESH_MS_DEFAULT, LOGO
from src.panel.control import start_worker, stop_worker, get_pid
from src.panel.helpers import leer_eventos, recientes, ultimo_evento

st.set_page_config(page_title="Neuromech Vision | Panel", page_icon="🧠", layout="wide")

THEME = """
<style>
:root { --bg:#0b0e12; --card:rgba(255,255,255,0.06); --border:rgba(255,255,255,0.12);
        --text:#e8eef6; --muted:#9aa4b2; --ok:#22c55e; --danger:#ef4444; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #131a24 0%, #0b0e12 40%), var(--bg); color: var(--text); }
.block-container { padding-top:.8rem; padding-bottom:2rem; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border:1px solid var(--border); border-radius:16px; padding:16px; }
.badge { padding:4px 10px; border-radius:999px; font-size:.8rem; border:1px solid var(--border); }
.live { background:#09351f; color:#34d399; border-color:#065f46; }
.off  { background:#3f1d1d; color:#f87171; border-color:#7f1d1d; }
.kpi { font-size:1.6rem; font-weight:800; }
.kpi-label { font-size:.9rem; color:var(--muted); margin-top:-4px; }
.timeline { display:flex; gap:12px; overflow-x:auto; padding: 6px 2px; }
.tile { min-width:160px; background: var(--card); border:1px solid var(--border); border-radius:12px; padding:8px; }
.tile img { border-radius:10px; border:1px solid var(--border); }
.footer { color: var(--muted); font-size:.85rem; }
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

RUN_DIR = pathlib.Path("data/run")
# Usamos 'data' directamente en lugar de 'RUN_DIR' para las fotos
LAST = pathlib.Path("data/last_frame.jpg")
PREV = pathlib.Path("data/last_frame_prev.jpg")
# -----------------------
STATUS = RUN_DIR / "vision.status"
EVENTS = pathlib.Path("data/exports.csv")
PIDFILE = RUN_DIR / "panel.pid"

def read_status():
    try: return STATUS.read_text(encoding="utf-8")
    except: return "cam=None backend=None size=0x0"

def parse_status(s: str):
    parts = {k:v for k,v in (p.split("=",1) for p in s.split() if "=" in p)}
    backend = parts.get("backend","-"); size = parts.get("size","-")
    ok = (size not in ("0x0","-")) and (backend not in ("None","-"))
    return ok, backend, size

def get_frame_path():
    for p in [LAST, PREV]:
        if p.exists():
            st_size = p.stat().st_size
            age = time.time() - p.stat().st_mtime
            if st_size >= 5000:
                if age < 0.18: time.sleep(0.18)
                return str(p)
    return None

# Sidebar
with st.sidebar:
    st.markdown("### Neuromech Vision")
    st.caption("Control de fuente y estado del productor")
    prefer = st.radio("Estrategia", ["auto","url","local"], index=2, horizontal=True)
    url = st.text_input("URL (si 'url' o 'auto')", os.getenv("CAM_URL","").strip(), placeholder="http://IP:4747/mjpegfeed?640x480")
    cam = st.number_input("Índice cámara (si 'local' o 'auto')", min_value=0, step=1, value=2)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Aplicar", use_container_width=True):
            stop_worker(PIDFILE); start_worker(PIDFILE, prefer=prefer, url=url, cam_idx=int(cam)); st.success("Productor aplicado")
    with c2:
        if st.button("Reiniciar", use_container_width=True):
            stop_worker(PIDFILE); start_worker(PIDFILE, prefer=prefer, url=url, cam_idx=int(cam)); st.info("Productor reiniciado")
    with c3:
        if st.button("Detener", use_container_width=True):
            stop_worker(PIDFILE); st.warning("Productor detenido")

# Autolanzar si no hay PID
if not get_pid(PIDFILE):
    start_worker(PIDFILE, prefer=prefer, url=url, cam_idx=int(cam))

# Header
col_logo, col_title = st.columns([1,6])
with col_logo:
    if LOGO.exists(): st.image(str(LOGO), width=56)
with col_title:
    st.markdown(f"## {APP_TITLE}")
    st.caption(APP_SUBTITLE)

ok, backend, size = parse_status(read_status())

# KPIs
def kpi_row():
    col1, col2, col3, col4 = st.columns([1.3,1,1,1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<span class="badge {"live" if ok else "off"}>{"En vivo" if ok else "Sin señal"}</span>', unsafe_allow_html=True)
        st.caption(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><div class="kpi">{}</div><div class="kpi-label">Backend</div></div>'.format(backend), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><div class="kpi">{}</div><div class="kpi-label">Resolución</div></div>'.format(size), unsafe_allow_html=True)
    with col4:
        host = socket.gethostbyname(socket.gethostname())
        st.markdown('<div class="card"><div class="kpi">localhost</div><div class="kpi-label">URL: http://localhost:8581</div></div>', unsafe_allow_html=True)
        st.caption(f"LAN: http://{host}:8581")
kpi_row()
st.divider()

# Tabs
tab_live, tab_id, tab_events, tab_diag = st.tabs(["En vivo", "Identidad", "Eventos", "Diagnóstico"])

with tab_live:
    col_live, col_side =st.columns([3.2,1.8])
    with col_live:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        #--- Inicio del cambio
        image_data = None
        if ok:
            path_str =  get_frame_path()
            if path_str:
                try:
                    # Al leer los bytes (.read_bytes()), Streamlit entiende que es 
                    # una imagen NUEVA y la actualiza obligatoriamente.
                    image_data = pathlib.Path(path_str).read_bytes()    
                except Exception:
                    # Si el archivo se está escribiendo justo ahora, ignoramos el error
                    pass
        if image_data:
            st.image(image_data, use_container_width=True)
        else:
            st.info("Esperando frames del productor o revisa la fuente en la barra lateral.")
        #--- Fin del cambio
        st.markdown("</div>", unsafe_allow_html=True)
     
     
     
    with col_side:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Resumen instantáneo")
        st.markdown(f'<span class="badge {"live" if ok else "off"}>{"En vivo" if ok else "Sin señal"}</span>', unsafe_allow_html=True)
        st.write("Backend:", backend); st.write("Resolución:", size)
        st.write("Hora:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.markdown("</div>", unsafe_allow_html=True)
        # Timeline simple
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Últimos reconocidos")
        df = leer_eventos(EVENTS)
        if not df.empty:
            rec = recientes(df, 8)
            cA, cB = st.columns(2)
            for i, row in rec.iterrows():
                target = cA if (i % 2 == 0) else cB
                with target:
                    snap = row.get("snapshot_path","")
                    if isinstance(snap,str) and len(snap)>0 and pathlib.Path(snap).exists():
                        st.image(snap, use_container_width=True)
                    nm = row.get("name","") or "Desconocido"
                    dec = row.get("decision","rejected")
                    st.write(nm, "🟢" if dec=="accepted" else "🔴")
                    st.caption(str(row.get("timestamp","")))
        else:
            st.caption("Aún no hay eventos.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_id:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Identidad predominante")
    df = leer_eventos(EVENTS)
    last = ultimo_evento(df)
    if last:
        cols = st.columns([1.2,2])
        with cols[0]:
            sp = last.get("snapshot_path","")
            if isinstance(sp,str) and len(sp)>0 and pathlib.Path(sp).exists():
                st.image(sp, use_container_width=True)
        with cols[1]:
            nm = last.get("name","") or "Desconocido"
            dec = last.get("decision","rejected")
            st.markdown(f"### {nm} {'🟢' if dec=='accepted' else '🔴'}")
            st.write("Código:", last.get("codigo",""))
            st.write("Grado:", last.get("grado",""))
            st.write("Confianza:", last.get("distancia",""))
    else:
        st.caption("Aún no hay identificaciones.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_events:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Eventos")
    df = leer_eventos(EVENTS)
    if not df.empty:
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            q = st.text_input("Filtrar por nombre o código", "")
            if q:
                df = df[df["name"].fillna("").str.contains(q, case=False) | df["codigo"].fillna("").str.contains(q, case=False)]
        with c2:
            estado = st.selectbox("Estado", ["Todos","accepted","rejected"])
            if estado != "Todos": df = df[df["decision"] == estado]
        with c3:
            quality = st.selectbox("Quality", ["Todas","high","mid"])
            if quality != "Todas": df = df[df["quality"] == quality]
        st.dataframe(df, use_container_width=True, height=420)
        st.download_button("Descargar CSV", data=EVENTS.read_bytes(), file_name="events.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Aún no hay eventos.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_diag:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Diagnóstico")
    st.write("Run dir:", RUN_DIR.resolve())
    st.write("PID productor:", get_pid(PIDFILE))
    st.code(read_status())
    st.write("Archivos:", str(LAST.resolve()), str(PREV.resolve()))
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.markdown('<div class="footer">Neuromech Labs • Panel con Streamlit • Accede vía localhost; 0.0.0.0 es solo dirección de enlace.</div>', unsafe_allow_html=True)

time.sleep(REFRESH_MS_DEFAULT/1000.0)
st.rerun()
