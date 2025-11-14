import ollama
import pandas as pd
from pathlib import Path
import datetime
import sys

# --- CONFIGURACIÓN ---
# Ajusta esta ruta si tu CSV está en otro lado (ej: data/exports.csv)
EVENTS_CSV = Path("data/logs/events.csv") 

# ¡IMPORTANTE! Pon aquí el nombre exacto del modelo que tienes en Ollama.
# Si descargaste llama3.2 usa "llama3.2". Si tienes gemma, pon "gemma:2b"
MODELO = "llama3.2" 

def generar_resumen_diario():
    # 1. Verificar si hay datos
    if not EVENTS_CSV.exists():
        print(f" No encuentro el archivo de eventos en: {EVENTS_CSV}")
        print("Asegurate de que el sistema de reconocimiento haya guardado algo hoy.")
        return

    try:
        df = pd.read_csv(EVENTS_CSV)
    except Exception as e:
        print(f" Error leyendo el CSV: {e}")
        return

    # 2. Filtrar eventos de HOY
    hoy = datetime.datetime.now().strftime("%Y-%m-%d")
    # Convertimos la columna timestamp a string por seguridad
    df['timestamp'] = df['timestamp'].astype(str)
    df_hoy = df[df['timestamp'].str.contains(hoy, na=False)]
    
    if df_hoy.empty:
        print(f" No hay registros con fecha de hoy ({hoy}).")
        return

    # 3. Calcular Estadísticas
    total = len(df_hoy)
    # Contamos 'rejected' como desconocidos
    desconocidos = df_hoy[df_hoy['decision'] == 'rejected'].shape[0]
    # Contamos 'accepted' como conocidos
    conocidos = df_hoy[df_hoy['decision'] == 'accepted'].shape[0]
    
    # Obtener lista de nombres únicos (sin repetir y quitando vacíos)
    nombres_vistos = df_hoy[df_hoy['decision'] == 'accepted']['name'].unique()
    nombres_vistos = [n for n in nombres_vistos if str(n) != 'nan']
    lista_nombres = ", ".join(nombres_vistos) if len(nombres_vistos) > 0 else "Ninguno"

    print(f" Analizando {total} eventos de hoy...")
    print(f"   - Conocidos: {conocidos}")
    print(f"   - Desconocidos/Intrusos: {desconocidos}")

    # 4. El Prompt para la IA
    prompt = f"""
    Eres el Jefe de Seguridad Inteligente de un colegio en Sitionuevo, Magdalena.
    Tu trabajo es escribir un reporte diario formal y breve basado en estos datos:
    
    FECHA: {hoy}
    TOTAL DETECCIONES: {total}
    PERSONAS AUTORIZADAS: {lista_nombres}
    ALERTA INTRUSOS (Desconocidos): {desconocidos}
    
    Instrucciones:
    - Escribe un solo párrafo resumen.
    - Si hay intrusos (desconocidos > 0), lanza una advertencia de seguridad.
    - Si todo es normal, confirma que el perímetro es seguro.
    - Usa un tono profesional y técnico.
    - No inventes nombres que no estén en la lista.
    """

    print(f"NeuroMech está pensando... (Usando modelo: {MODELO})")
    
    try:
        # Enviamos la orden a Ollama
        response = ollama.chat(model=MODELO, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        print("\n" + "="*40)
        print("REPORTE DE SEGURIDAD NEUROMECH")
        print("="*40)
        print(response['message']['content'])
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"\n Error conectando con Ollama: {e}")
        print("Sugerencia: ¿Está abierta la aplicación de Ollama? ¿Descargaste el modelo?")

if __name__ == "__main__":
    generar_resumen_diario()