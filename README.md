# empathic-zoom

## 1 - Descargar el archivo
Debe descargar el archivo fer2013.csv con el link a continuacion:
https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition?select=fer2013.csv y dejarno en el directorio `./input/`

## 2 - Instalación de librerias

## 3 - Entrenar el modelo CNN
Este paso puede saltearse ya que en los archivos esta en modelo CNN previamente entrenado (modelCNN.joblib)

## 4 - Ejecutar el script de conexion a zoom y deteccion de emociones
Al final del script `conectZoomAndPredictEmotion.py` debe colocar los datos de la reunion:
   * id_reunion = 'XXX'
   * contrasenia_reunion = 'YYY'
   * name_zoom = "Usuario Empathic Zoom"
   * capture_interval = 2 (intervalo en segundo en que se calculan las emociones)