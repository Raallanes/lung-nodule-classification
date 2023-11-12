import os
import cv2
import pydicom
import PySimpleGUI as sg

# Utils
from utils import normalize_min_max

# Utils Image
from utils_image import image_scale, show_image, kurita_binarization, kittler_illingworth_threshold, image_roi_select

# UI
from gui.browsers import images_explorer;
from gui import config_ui;


# Config UI
config_ui

images = images_explorer()
for image_path in images:
    # Cargar la imagen DICOM
    dicom_file = pydicom.dcmread(image_path)
    image = dicom_file.pixel_array

    # Definir la carpeta de destino y el nombre del archivo DICOM de la ROI
    output_folder = './assets/output'
    image_name = image_path.split('/')[-1]
    output_filename = "test_" + image_name

    # Normalizar imagen
    image_normalized = normalize_min_max(image)

    # Escalar el contraste de la imagen
    image_scalated = image_scale(image)

    #Muestra la imagen original
    show_image('Imagen original', image_scalated)

    # Aplicar el método de Kurita a la imagen
    image_binary = kurita_binarization(image_normalized)
    
    # Preguntar al usuario si la binarización es correcta  
    def ask_binarization():
        global user_answer
        user_answer = sg.popup_yes_no("¿La binarización es correcta?")
    
    # Mostrar la imagen binarizada
    show_image('Imagen binarizada Metodo Kurita', image_binary, ask_binarization)

    if user_answer == 'No':
        sg.popup("Vamos a intentar otro método de binarización.")
        image_binary = kittler_illingworth_threshold(image_scalated)
        show_image('Imagen binarizada Metodo Kittler', image_binary)
    else:
        sg.popup("Excelente, continuemos con el programa.")

    cv2.destroyAllWindows()

    # Seleccionar roi de la imagen
    image_selected = image_roi_select(image_binary, dicom_file)

    # Guardad la selección de la imagen
    if image_selected is not None:
        # Guardar el archivo DICOM modificado con la ROI
        output_path = os.path.join(output_folder, output_filename)
        image_selected.save_as(output_path)
        sg.popup("Imagen DICOM con ROI guardada en ", output_path)
    else:
         sg.popup("No se ha seleccionado una región de interés (ROI).")
