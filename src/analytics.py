import numpy as np
import cv2

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

def get_head_pose(shape, w, h):
    """
    Calcula si la persona está mirando al frente (Atenta) o distraída
    basado en la geometría de la nariz y los ojos.
    """
    # Puntos clave 2D (Imagen)
    image_points = np.array([
        shape[30],     # Nariz
        shape[8],      # Mentón
        shape[36],     # Ojo Izquierdo (esquina)
        shape[45],     # Ojo Derecho (esquina)
        shape[48],     # Boca Izquierda
        shape[54]      # Boca Derecha
    ], dtype="double")

    # Puntos clave 3D (Modelo Genérico Humano)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nariz
        (0.0, -330.0, -65.0),        # Mentón
        (-225.0, 170.0, -135.0),     # Ojo Izquierdo
        (225.0, 170.0, -135.0),      # Ojo Derecho
        (-150.0, -150.0, -125.0),    # Boca Izquierda
        (150.0, -150.0, -125.0)      # Boca Derecha
    ])

    # Datos internos de cámara simulados
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double")
    
    dist_coeffs = np.zeros((4, 1)) # Asumimos sin distorsión de lente

    # Resolver PnP (Perspectiva de n Puntos)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Proyectar la "nariz" hacia adelante para ver a dónde apunta
    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )

    p1 = (int(image_points[0][0]), int(image_points[0][1])) # Punta nariz real
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])) # A dónde apunta

    # Lógica de Atención (Si apunta muy lejos del centro, está distraído)
    # Ángulos simples basados en la diferencia de coordenadas
    diff_x = p2[0] - p1[0]
    diff_y = p2[1] - p1[1]

    status = "ATENTO"
    color = (0, 255, 0) # Verde

    # Umbrales de distracción (Ajustar según pruebas en Sitionuevo)
    if abs(diff_x) > 150:
        status = "DISTRAIDO (Lado)"
        color = (0, 0, 255) # Rojo
    elif diff_y > 100: # Mirando abajo (Celular?)
        status = "DISTRAIDO (Abajo)"
        color = (0, 0, 255)
    elif diff_y < -100:
        status = "DISTRAIDO (Arriba)"
        color = (0, 0, 255)

    return status, color, p1

def get_emotion(name, face_img):
    """
    Analiza la emoción usando DeepFace (si está instalado).
    """
    if not HAS_DEEPFACE:
        return "N/A (Instalar deepface)"
    
    # Filtro: Si la imagen es muy pequeña, no analizar (ahorra CPU)
    if face_img.shape[0] < 40 or face_img.shape[1] < 40:
        return "-"

    try:
        # enforce_detection=False es vital para caras de lado
        objs = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, verbose=False)
        if len(objs) > 0:
            dominant = objs[0]['dominant_emotion']
            
            # Traductor simple
            traduccion = {
                "happy": "FELIZ", "sad": "TRISTE", "angry": "ENOJADO", 
                "neutral": "NEUTRAL", "fear": "MIEDO", "surprise": "SORPRESA", 
                "disgust": "DISGUSTO"
            }
            return traduccion.get(dominant, dominant.upper())
    except Exception:
        pass
    
    return "-"