import os
import csv

# Ruta base del dataset
base_dir = os.path.join("data", "dataset")
os.makedirs(base_dir, exist_ok=True)

# Lista de estudiantes
estudiantes = [
    {"codigo": "10001", "nombre": "Aldis", "apellido": "Cardenas", "grado": "9-2"},
    {"codigo": "10002", "nombre": "Brian", "apellido": "Manga", "grado": "6-1"},
    {"codigo": "10003", "nombre": "Victor", "apellido": "Torres", "grado": "6-1"},
    {"codigo": "10004", "nombre": "Adaines", "apellido": "Torres", "grado": "9-2"},
    {"codigo": "10005", "nombre": "Jerson", "apellido": "Almarales", "grado": "9-2"},
    {"codigo": "10006", "nombre": "Jhon", "apellido": "Monsalvo", "grado": "9-2"},
    {"codigo": "10007", "nombre": "Maria", "apellido": "Mendoza", "grado": "9-6"},
    {"codigo": "10008", "nombre": "Julian", "apellido": "Torres", "grado": "9-3"},
    {"codigo": "10009", "nombre": "Yenis", "apellido": "Gutierrez", "grado": "9-3"},
    {"codigo": "10010", "nombre": "Santiago", "apellido": "Silvera", "grado": "9-3"},
    {"codigo": "10011", "nombre": "Aislin", "apellido": "Pautt", "grado": "9-4"},
    {"codigo": "10012", "nombre": "Liceth", "apellido": "Rodriguez", "grado": "9-3"},
    {"codigo": "10013", "nombre": "Jhon Janer", "apellido": "Merino", "grado": "9-4"},
    {"codigo": "10014", "nombre": "Juan", "apellido": "Diaz", "grado": "9-4"},
    {"codigo": "10015", "nombre": "Santiago", "apellido": "Monsalvo", "grado": "9-4"},
]

# Crear carpetas y archivo CSV
csv_path = os.path.join("data", "estudiantes.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["codigo", "nombre", "apellido", "grado", "ruta_carpeta"])
    
    for est in estudiantes:
        carpeta = f"{est['codigo']}_{est['nombre']}_{est['apellido']}".replace(" ", "_")
        ruta_estudiante = os.path.join(base_dir, carpeta)
        os.makedirs(ruta_estudiante, exist_ok=True)
        writer.writerow([est['codigo'], est['nombre'], est['apellido'], est['grado'], ruta_estudiante])
        print(f"✅ Carpeta creada: {ruta_estudiante}")

print("\n📄 Archivo CSV creado en:", os.path.abspath(csv_path))
print("📸 Ahora puedes colocar manualmente las fotos (1.jpg, 2.jpg, 3.jpg) en cada carpeta.")
