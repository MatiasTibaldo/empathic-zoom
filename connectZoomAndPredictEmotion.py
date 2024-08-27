from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cv2
import dlib
from joblib import load
import numpy as np
import time
from PIL import Image
import io

# Configuración del modelo de emociones
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
joblib_filename = './modelCNN.joblib'  # Archivo del modelo entrenado
model = load(joblib_filename)

# Configuración del detector de caras
detector = dlib.get_frontal_face_detector()
FACE_SHAPE = (200, 200)  # Tamaño del frame de captura

def get_emotion_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    faces = detector(clahe_image)
    emotions = []
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = clahe_image[y:y+h, x:x+w]
        
        if face_image.size != 0:
            face_image_resized = cv2.resize(face_image, (46, 46))
            image_array = np.expand_dims(face_image_resized, axis=-1) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            predictions = model.predict(image_array)
            emotion = emotion_map[np.argmax(predictions)]
            emotions.append((x, y, w, h, emotion))
    return emotions

def main():
    # Configuración del navegador
    service = ChromeService(executable_path='/usr/bin/chromedriver')
    driver = webdriver.Chrome(service=service)
    driver.get('https://app.zoom.us/wc/' + id_reunion + '/join')
    time.sleep(5)

    driver.find_element(By.ID, 'input-for-pwd').send_keys(contrasenia_reunion)
    driver.find_element(By.ID, 'input-for-name').send_keys(name_zoom)
    driver.find_element(By.CLASS_NAME, 'zm-btn').click()
    time.sleep(20)

    # Esperar a que la vista de galería esté disponible
    gallery_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, 'gallery-video-container__main-view'))
    )
    
    # Intervalo para tomar capturas de pantalla (en segundos)
    last_capture_time = time.time()
    
    while True:
         # Tomar una captura de pantalla cada `capture_interval` segundos
        if time.time() - last_capture_time > capture_interval:
            last_capture_time = time.time()
            # Captura la vista de galería de Zoom
            png_screenshot = driver.get_screenshot_as_png()

            # Convertir el objeto PNG a una imagen PIL
            image = Image.open(io.BytesIO(png_screenshot))

            # Convertir la imagen PIL a un array de NumPy
            image_np = np.array(image)

            # Convertir la imagen RGB a BGR (OpenCV usa BGR por defecto)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Procesar la captura para detectar emociones
            emotions = get_emotion_from_image(image_bgr)
            print(emotions)
            # Mostrar rectángulos y etiquetas en la imagen capturada
            for (x, y, w, h, emotion) in emotions:
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image_bgr, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar el frame con las anotaciones
            cv2.imshow("Emotion Detection from Screenshot", image_bgr)



        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cv2.destroyAllWindows()
    driver.quit()

if __name__ == "__main__":
    # Parámetros de la reunión
    id_reunion = '76332965308'
    contrasenia_reunion = 'iVMqSxJ9OI6NrdGfpsHHs7pDcIp7gk.1'
    name_zoom = "Matias Test"
    capture_interval = 10  # Intervalo en segundos para tomar capturas de pantalla
    main()
