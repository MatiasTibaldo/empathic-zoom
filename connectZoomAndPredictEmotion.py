import tkinter as tk
from collections import Counter
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
from PIL import Image, ImageTk
import io
import threading

# Configuración del modelo de emociones (en español)
emotion_map = {0: 'Enojado', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
joblib_filename = './modelCNN.joblib'  # Archivo del modelo entrenado
model = load(joblib_filename)

# Configuración del detector de caras
detector = dlib.get_frontal_face_detector()
FACE_SHAPE = (200, 200)  # Tamaño del frame de captura

# Crear la ventana de la interfaz gráfica
root = tk.Tk()
root.title("Resumen de emociones")

# Configurar tamaño fijo y fondo de la ventana
root.geometry("400x300")
root.resizable(False, False)
root.config(bg="lightblue")  # Cambiar el color de fondo

# Diccionario para cargar emojis correspondientes a las emociones
emoji_paths = {
    'Enojado': './emojis/Enojado.png',
    'Disgusto': './emojis/Disgusto.png',
    'Miedo': './emojis/Miedo.png',
    'Feliz': './emojis/Feliz.png',
    'Triste': './emojis/Triste.png',
    'Sorpresa': './emojis/Sorpresa.png',
    'Neutral': './emojis/Neutral.png',
}

# Cargar los emojis y guardarlos en un diccionario de imágenes
emoji_images = {emotion: ImageTk.PhotoImage(Image.open(path).resize((30, 30))) for emotion, path in emoji_paths.items()}

# Etiqueta inicial
summary_frame = tk.Frame(root, bg="lightblue")
summary_frame.pack(pady=20)

# Función para actualizar la interfaz gráfica con el conteo de emociones
def update_emotion_summary(emotion_counts):
    # Eliminar los widgets previos
    for widget in summary_frame.winfo_children():
        widget.destroy()

    # Crear una fila por cada emoción
    for emotion, count in emotion_counts.items():
        frame = tk.Frame(summary_frame, bg="lightblue")
        frame.pack(anchor="w")

        # Colocar el emoji de la emoción
        emoji_label = tk.Label(frame, image=emoji_images[emotion], bg="lightblue")
        emoji_label.pack(side="left", padx=5)

        # Colocar el texto con la emoción y el conteo
        text_label = tk.Label(frame, text=f"{emotion}: {count}", font=("Helvetica", 14), bg="lightblue")
        text_label.pack(side="left")

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

def emotion_detection_loop():
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
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, 'gallery-video-container__main-view'))
    )
    
    # Intervalo para tomar capturas de pantalla (en segundos)
    last_capture_time = time.time()
    
    while True:
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
            
            # Contar las emociones detectadas
            emotion_list = [emotion for _, _, _, _, emotion in emotions]
            emotion_counts = dict(Counter(emotion_list))
            
            print(emotion_counts)

            # Actualizar la interfaz gráfica con el resumen de emociones
            update_emotion_summary(emotion_counts)

            # Mostrar rectángulos y etiquetas en la imagen capturada
            for (x, y, w, h, emotion) in emotions:
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image_bgr, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar el frame con las anotaciones
            ## DESCOMENTAR ESTA LINEA SI SE QUIERE VER LAS EMOCIONES EN LA CAPTURA DE LOS PARTICIPANTES
            # cv2.imshow("Emotion Detection from Screenshot", image_bgr)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cv2.destroyAllWindows()
    driver.quit()

if __name__ == "__main__":
    # Parámetros de la reunión
    id_reunion = '73863024317'
    contrasenia_reunion = 'X6mBiHUad2gY3Nj96lrXBLsVb2fwHv.1'
    name_zoom = "Matias Test"
    # Intervalo en segundos para tomar capturas de pantalla
    capture_interval = 2  

    # Ejecutar la captura de emociones en un hilo separado
    thread = threading.Thread(target=emotion_detection_loop)
    thread.start()

    # Iniciar el loop principal de Tkinter
    root.mainloop()
