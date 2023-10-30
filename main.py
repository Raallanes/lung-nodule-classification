''''
    Fichero Principal
'''
import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

from Functions import kurita_binarization
from Functions import kittler_illingworth_threshold
from Functions import apply_roi
from Functions import apply_roi_to_dicom
from Functions import show_image

# Cargar la imagen DICOM
dicom_file = pydicom.dcmread('D:\Tesis\DataBase Rachel\Data DICOM\P6-063.dcm')
I = dicom_file.pixel_array
min_value = np.min(I)
normalized_image = (I - min_value) / (np.max(I) - min_value)

# Definir la carpeta de destino y el nombre del archivo DICOM de la ROI
output_folder = 'D:\Tesis\DataBase Rachel\Metodo Kittler'
output_filename = 'A.dcm'

nbins = 256
k = 1.0
strel1 = 100
strel2 = 200

# Obtener el tamaño de la imagen (M y N)
M, N = I.shape  # Suponiendo que 'I' es una matriz NumPy que representa la imagen

# Determinar la clase de 'I' y asignar el valor correspondiente a 'L'
# Aquí suponemos que 'I' es una matriz NumPy que puede ser de tipo float o entero.
ImClas = I.dtype
if ImClas == np.float64:
    L = 1
elif ImClas == np.uint8:
    L = 255
elif ImClas == np.uint16 or ImClas == np.int16:
    L = 2**16

# Ajustar el contraste entre 0 y 2000
I_contrast = np.clip(I, 0, 2000)  # Limita los valores entre 0 y 2000

# Escala los valores a 8 bits (0-255) para mostrar la imagen
I_scaled = cv2.convertScaleAbs(I_contrast, alpha=(255.0/2000.0))

#Muestra la imagen original
show_image('Imagen original',I_scaled)

# Aplicar el método de Kurita a la imagen
binary_image, umbral1 = kurita_binarization(normalized_image,nbins,k)

#Mostrar la imagen binarizada
show_image('Imagen binarizada Metodo Kurita',binary_image)


# Preguntar al usuario si la binarización es correcta
user_input = input("¿La binarización es correcta? (s/n): ")

if user_input.lower() == 'n':
    print("Vamos a intentar otro método de binarización.")
    binary_image = kittler_illingworth_threshold(I_scaled)
    show_image('Imagen binarizada Metodo Kittler',binary_image)
else:
    print("Excelente, continuemos con el programa.")

cv2.destroyAllWindows()

# Crear una figura para mostrar la imagen binarizada
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(binary_image, cmap='gray')

binary_image_roi = []
# Inicializar la selección de polígonos en la figura
vertices = []
polygon_selector = PolygonSelector(ax, onselect=None)

# Conectar la función para aplicar la ROI después de seleccionarla
polygon_selector.onselect = lambda vertices: apply_roi(binary_image, vertices, ax,binary_image_roi)

plt.show()

# Cuando la ROI está lista y deseas guardarla en el archivo DICOM original
if binary_image_roi:
    binary_image_roi[0] = (binary_image_roi[0].astype(np.float32)/255.0*65535.0-32768.0).astype(np.int16)
    dicom_file_with_roi = apply_roi_to_dicom(dicom_file,binary_image_roi[0])

    if dicom_file_with_roi is not None:
        # Guardar el archivo DICOM modificado con la ROI
        output_path = os.path.join(output_folder, output_filename)
        dicom_file_with_roi.save_as(output_path)

        print("Imagen DICOM con ROI guardada en", output_path)
    else:
        print("No se ha seleccionado una región de interés (ROI).")
else:
    print("No se ha seleccionado una región de interés (ROI).")

