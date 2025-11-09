# src/repositories.py
import csv
from typing import Dict, Optional

class StudentRepository:
    def get(self, codigo: str) -> Optional[Dict]:
        raise NotImplementedError

class CSVStudentRepository(StudentRepository):
    def __init__(self, csv_path: str):
        self._data = {}
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                codigo = row["codigo"]
                self._data[codigo] = {
                    "nombre": row.get("nombre", ""),
                    "apellido": row.get("apellido", ""),
                    "grado": row.get("grado", ""),
                    "ruta": row.get("ruta_carpeta", ""),
                }

    def get(self, codigo: str) -> Optional[Dict]:
        return self._data.get(codigo)

class DBStudentRepository(StudentRepository):
    def __init__(self, dsn: str):
        # dsn: cadena de conexión (PostgreSQL/MySQL/SQL Server etc.)
        self._dsn = dsn
        # TODO: inicializa pool/conexión aquí

    def get(self, codigo: str) -> Optional[Dict]:
        # TODO: consulta SQL real
        # SELECT nombre, apellido, grado FROM estudiantes WHERE codigo = %s
        return None
