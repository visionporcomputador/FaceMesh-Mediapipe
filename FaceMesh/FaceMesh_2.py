"""
DIBUJO DE CADA UNO DE LOS PUNTOS DE REFERENCIA

No se hace uso de funcion mp_drawing.draw_landmarks para dibujar.
Se recorre la lista de los 468 puntos clave del facemesh,
y se dibuja circulo de un color particular.


face_landmarks.landmark : lista de 468 tuplas [x,y,z]
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh



drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("No se realizo lectura")
      continue

    # Reflejo horizontal - Conversion BGR a RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # Para mejorar rendimiento, marcar imagen coomo no escribible
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Dibujar puntos de referencia
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:

        print(len(face_landmarks.landmark)) #468 - esta es la lista de puntos de referencia

        """
        - face_landmarks.landmark[0]:    me genera las coordenadas x.y.z del punto
                                         de referencia con indice 0 /valores entre 0-1
        - face_landmarks.landmark[0].x : me genera la coordenada x del punto
                                         de referencia con indice 0 / valores entre 0-1

        """
        print(face_landmarks.landmark[0]) #punto -0- de referencia

        # Recorrer cada punto de referncia 468 puntos
        # normalizar valor de coordenada y dibujar circulo
        for landmark in face_landmarks.landmark:
            x = landmark.x
            y = landmark.y

            shape = image.shape 
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])

            cv2.circle(image, (relative_x,relative_y), 1, (255,255,0), cv2.FILLED, cv2.LINE_AA)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
cap.release()
cv2.destroyAllWindows()
