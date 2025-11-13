import cv2
import os
import shutil
from pathlib import Path

# --- CONFIGURACIÓN ---
SNAPSHOTS_DIR = Path("data/snapshots")
DATASET_DIR = Path("data/dataset")
# ---------------------

def safe_name(name):
    return "".join([c for c in name if c.isalnum() or c in (' ','_')]).strip().replace(' ','_')

def find_existing_folder(partial_name, base_path):
    """Busca una carpeta que contenga el nombre, para respetar los IDs (ej: 10001_Nombre)"""
    if not base_path.exists(): return None
    
    # Normalizamos para comparar (minusculas, sin espacios)
    search = partial_name.lower().replace("_", " ")
    
    for folder in os.listdir(base_path):
        if os.path.isdir(base_path / folder):
            # Comparamos si el nombre buscado está dentro del nombre de la carpeta
            folder_norm = folder.lower().replace("_", " ")
            if search in folder_norm:
                return base_path / folder
    return None

def main():
    if not SNAPSHOTS_DIR.exists():
        print(f"No existe la carpeta {SNAPSHOTS_DIR}")
        return

    # Filtramos solo imagenes
    files = [f for f in os.listdir(SNAPSHOTS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print("¡No hay snapshots nuevos para revisar!")
        return

    print(f"--- INICIANDO REVISIÓN DE {len(files)} FOTOS ---")
    print("TECLAS DE CONTROL:")
    print(" [ENTER] : Confirmar (Mueve a la carpeta detectada).")
    print(" [n]     : Corregir nombre manualmente.")
    print(" [d]     : Borrar foto (mala calidad/basura).")
    print(" [q]     : Salir.")
    print("-" * 40)

    count = 0
    for filename in files:
        file_path = SNAPSHOTS_DIR / filename
        
        # Intentamos extraer el nombre del archivo del snapshot
        # Formato esperado: YYYYMMDD_HHMMSS_Nombre_Apellido.jpg
        parts = filename.split('_')
        predicted_name = "Desconocido"
        
        # Si tiene mas de 2 partes, asumimos que lo que sigue a la fecha/hora es el nombre
        if len(parts) >= 3:
            raw_name = "_".join(parts[2:]).replace(".jpg","").replace(".png","")
            predicted_name = raw_name
        
        # BÚSQUEDA INTELIGENTE DE CARPETA
        existing_folder = find_existing_folder(predicted_name, DATASET_DIR)
        
        # Si encontramos la carpeta con ID (ej: 10001_Aldis...), sugerimos esa
        if existing_folder:
            target_folder_name = existing_folder.name
            is_new_folder = False
        else:
            target_folder_name = predicted_name
            is_new_folder = True

        # Cargar y mostrar imagen
        img = cv2.imread(str(file_path))
        if img is None: continue
        display_img = cv2.resize(img, (0,0), fx=2.0, fy=2.0)
        
        # Agregar texto informativo en la imagen (opcional)
        cv2.putText(display_img, f"Sugerencia: {target_folder_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Revisor (Presiona ENTER, D o N)", display_img)
        
        print(f"\nFoto: {filename}")
        print(f"--> Destino sugerido: [{target_folder_name}] {'(NUEVA)' if is_new_folder else '(EXISTENTE)'}")
        
        key = cv2.waitKey(0)

        final_folder_path = None

        if key == ord('q'): # Salir
            break
        
        elif key == ord('d'): # Delete
            os.remove(file_path)
            print(" Foto borrada.")
            continue
            
        elif key == ord('n'): # Rename
            new_name = input("Escribe el nombre o ID correcto: ").strip()
            # Buscamos de nuevo por si el usuario escribió el nombre real
            found = find_existing_folder(new_name, DATASET_DIR)
            if found:
                final_folder_path = found
            else:
                final_folder_path = DATASET_DIR / safe_name(new_name)
            
        elif key == 13 or key == ord('y'): # Enter
            if existing_folder:
                final_folder_path = existing_folder
            else:
                final_folder_path = DATASET_DIR / target_folder_name
            
        else:
            print("Tecla no reconocida, saltando...")
            continue

        # MOVER EL ARCHIVO
        if final_folder_path:
            final_folder_path.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(file_path), str(final_folder_path / filename))
                print(f" Movido a: {final_folder_path.name}")
                count += 1
            except Exception as e:
                print(f" Error moviendo archivo: {e}")

    cv2.destroyAllWindows()
    print(f"\nResumen: {count} fotos procesadas.")

if __name__ == "__main__":
    main()