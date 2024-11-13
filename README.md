
# empathic-zoom

El siguiente programa fue diseñado para conectarse a una reunión de zoom para predecir las emociones de los participantes que pueden verse en la vista galería. 
Al ejecutar el script se conectará automáticamente a la reunión de zoom a través del ID y CONTRASEÑA de la misma haciendo utilización de herramientas como `ChromeDriver` y `Selenium`. Luego se debe cambiar a la vista "GALERÍA" y el script a los pocos segundos comenzará a tomar capturas de la interfaz de la reunión para capturar y recortar los rostros presentes con la utilización de la librería `dlib`. Luego, calculará la emoción predominante en cada rostro utilizando un modelo construido previamente utilizando CNN (Convolutional Neural Network) y finalmente se expone en una sencilla interfaz visual la cantidad de personas en la reunión (de las cuales se pudo capturar su rostro) que están en determinado estado de ánimo. El modelo clasificatorio soporta 7 emociones que fueron determinadas a partir del entrenamiento, validación y testing de un dataset de rostros previamente clasificados denominado `fer2013`. Las emociones que clasifica son las siguientes:
- Enojado
- Disgusto
- Miedo
- Feliz
- Triste
- Sorpresa
- Neutral

## 1 - Descargar el archivo

Debe descargar el archivo fer2013.csv con el link a continuación:
https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition?select=fer2013.csv y dejarno en el directorio `./input/`

## 2 - Instalación de librerías
Se recomienda usar un enviroment local para el proyecto por la compatabilidad de librerias. En este proyecto se uso python 3.10.15

La instalacion de las librerias puede hacer de manera automatica corriendo el comando `pip install -r requirements.txt` o de manera manual instalando las librerias a continuacion:

pip install opencv-python
pip install cmake
pip install dlib
pip install joblib
pip install image 
pip install dill 
pip install keras
pip install tensorflow
pip install webdriver-manager
pip install tk
pip install selenium 

### Otras consideraciones
Dependiendo el sistema operativo utilizado (Windows) sera necesario instalar algunas librerias de manera manual:

https://github.com/z-mahmud22/Dlib_Windows_Python3.x
python -m pip install dlib-19.24.99-cp312-cp312-win_amd64.whl

https://www.selenium.dev/documentation/webdriver/troubleshooting/errors/driver_location/#download-the-driver

### Como usar el webdriver manager
https://github.com/SergeyPirogov/webdriver_manager?tab=readme-ov-file#use-with-chrome
pip install webdriver-manager

### Obtener el chromedriver
https://developer.chrome.com/docs/chromedriver/downloads?hl=es-419#chromedriver_1140573590

### Seteo del driver de chrome
Dependiendo donde se realizo la instalacion de chromedriver, es necesario cambiar el path en donde esta el ejecutable (linea 94 del archivo connectZoomAndPredictEmotion.py)
executable_path='C:\Selenium\chromedriver.exe'

## 3 - Entrenar el modelo CNN
Lo que se busca en este paso es generar el modelo CNN necesario para luego clasificar los nuevos inputs.
Este paso puede saltarse ya que en los archivos está en modelo CNN previamente entrenado (modelCNN.joblib)
Si desea correrlo debe ejecutar el archivo `CNN_modelo_FER.py`

## 4 - Ejecutar el script de conexion a zoom y detección de emociones

Al final del script `connectZoomAndPredictEmotion.py` debe colocar los datos de la reunión:

- id_reunion = 'XXX'
- contrasenia_reunion = 'YYY'
- name_zoom = "Usuario Empathic Zoom"
- capture_interval = 2 (intervalo en segundos en que se calculan las emociones)

A continuación se debe correr el script mencionado. 


