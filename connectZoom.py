from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time

# Funcion que permite unirse a la reunion de zoom
# Luego de unirse debe cambiar la vista a Galeria
# Y a los 20segundos toma una captura de pantalla de los participantes
def unirse_a_reunion(id_reunion, contraseña_reunion):
    # Ruta al controlador de ChromeDriver, descárgalo de https://sites.google.com/a/chromium.org/chromedriver/downloads
    service = webdriver.ChromeService(executable_path = '/usr/bin/chromedriver')
    driver = webdriver.Chrome(service=service)

    # Abrir Zoom e ingresar a la reunión
    driver.get('https://app.zoom.us/wc/' + id_reunion + '/join')
    time.sleep(5)  # Esperar a que la página cargue completamente

    # Ingresar la contraseña de la reunión
    meeting_password_input = driver.find_element(By.ID,'input-for-pwd')
    meeting_password_input.send_keys(contraseña_reunion)

    # Ingresar la contraseña de la reunión
    meeting_personal_name_input = driver.find_element(By.ID,'input-for-name')
    meeting_personal_name_input.send_keys("Matias Test")

    # Hacer clic en "Unirse"
    join_button = driver.find_element(By.CLASS_NAME, "zm-btn")
    join_button.click()
    time.sleep(20)

    galery = WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.CLASS_NAME, 'gallery-video-container__main-view')))
    galery.screenshot('gallery.png')
    # Esperar un tiempo suficiente para unirse a la reunión
    time.sleep(120)

    # Cerrar el navegador después de unirse a la reunión
    driver.quit()

# Ejemplo de uso
id_reunion = '86115514104'
contraseña_reunion = 'XCfFk4ZvI91bDZeQb19KazVJkHmzEy.1'
unirse_a_reunion(id_reunion, contraseña_reunion)

# https://us05web.zoom.us/j/86115514104?pwd=XCfFk4ZvI91bDZeQb19KazVJkHmzEy.1
