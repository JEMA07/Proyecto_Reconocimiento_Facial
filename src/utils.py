# Funciones auxiliares
import pickle

with open("models/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Total de rostros entrenados: {len(data['encodings'])}")
print(f"Nombres registrados: {set(data['names'])}")
