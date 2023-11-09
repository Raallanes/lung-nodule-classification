import os
import cv2
import pydicom

# Utils
from utils import normalize_min_max

# Utils Image
from utils_image import image_scale, show_image, kurita_binarization, kittler_illingworth_threshold, image_roi_select

# Cargar la imagen DICOM
dicom_file = pydicom.dcmread('D:/Tesis/DataBase Rachel/1. Data DICOM/1-050.dcm')
image = dicom_file.pixel_array

# Definir la carpeta de destino y el nombre del archivo DICOM de la ROI
output_folder = 'D:/Tesis/DataBase Rachel/Segmentacion Kurita Python'
output_filename = 'a.dcm'

# Normalizar imagen
image_normalized = normalize_min_max(image)

# Escalar el contraste de la imagen
image_scalated = image_scale(image)

#Muestra la imagen original
show_image('Imagen original', image_scalated)

# Aplicar el método de Kurita a la imagen
image_binary = kurita_binarization(image_normalized)

# Mostrar la imagen binarizada
show_image('Imagen binarizada Metodo Kurita', image_binary)

# Preguntar al usuario si la binarización es correcta
user_input = input("¿La binarización es correcta? (s/n): ")

if user_input.lower() == 'n':
    print("Vamos a intentar otro método de binarización.")
    image_binary = kittler_illingworth_threshold(image_scalated)
    show_image('Imagen binarizada Metodo Kittler', image_binary)
else:
    print("Excelente, continuemos con el programa.")

cv2.destroyAllWindows()

# Seleccionar roi de la imagen
image_selected = image_roi_select(image_binary, dicom_file)

# Guardad la selección de la imagen
if image_selected is not None:
    # Guardar el archivo DICOM modificado con la ROI
    output_path = os.path.join(output_folder, output_filename)
    image_selected.save_as(output_path)

    print("Imagen DICOM con ROI guardada en", output_path)
else:
    print("No se ha seleccionado una región de interés (ROI).")
