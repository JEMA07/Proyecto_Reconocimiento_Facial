import cv2

cap = cv2.VideoCapture(0)  # Usa el índice 0 (ya confirmado)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
else:
    print("✅ Cámara conectada. Mostrando video en vivo...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se recibe señal de video.")
            break
        cv2.imshow("Prueba - DroidCam Virtual Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
