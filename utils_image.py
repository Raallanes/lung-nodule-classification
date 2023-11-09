import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

from utils_roi import apply_roi, apply_roi_to_dicom


# Image utils

# Image Scale

def image_scale(image):
    # Ajustar el contraste entre 0 y 2000
    # Limita los valores entre 0 y 2000
    image_contrast = np.clip(image, 0, 2000)

    # Escala los valores a 8 bits (0-255) para mostrar la imagen
    return cv2.convertScaleAbs(image_contrast, alpha=(255.0 / 2000.0))


# Show Image
def show_image(text, img):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Segmentation

# Kurita Binarization

def kurita_binarization(image):
    # Paso 1: Inicializar variables
    threshold = np.mean(image)  # Estimación inicial del umbral
    prev_threshold = 0
    epsilon = 1e-5  # Criterio de convergencia

    # Paso 2: Iterar hasta que el umbral converja
    while abs(threshold - prev_threshold) > epsilon:
        foreground_pixels = image[image >= threshold]
        background_pixels = image[image < threshold]

        # Paso 3: Calcular la media de ambas clases
        mean_foreground = np.mean(foreground_pixels)
        mean_background = np.mean(background_pixels)

        # Paso 4: Actualizar el umbral
        prev_threshold = threshold
        threshold = 0.5 * (mean_foreground + mean_background)

    bwKur = (image > threshold).astype(np.uint8) * 255
    # Adjust threshold value based on the original data type
    if image.dtype == np.uint8:
        threshold = np.round(255 * threshold)
    elif image.dtype == np.uint16:
        threshold = np.round(65535 * threshold)

    return bwKur


# Kittler Illingworth Threshold

def kittler_illingworth_threshold(image):
    # Calcular el histograma de la imagen
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Número de bins
    nbins = len(hist)

    # Inicialización de variables
    p = np.zeros(nbins)
    omega = np.zeros(nbins)
    mu = np.zeros(nbins)
    inv_sigma2 = np.zeros(nbins)
    between_var = np.zeros(nbins)

    # Inicialización de umbrales
    best_threshold = 0
    max_between_var = 0

    # Calcular probabilidades y medias
    total_pixels = np.sum(hist)
    for i in range(nbins):
        p[i] = hist[i] / total_pixels
        omega[i] = omega[i - 1] + p[i]
        mu[i] = mu[i - 1] + i * p[i]

    # Calcular la inversa de las varianzas
    for i in range(nbins):
        if omega[i] > 0 and omega[i] < 1:
            inv_sigma2[i] = 1.0 / (p[i] * (1 - omega[i]))

    # Calcular entre-clase varianza
    for i in range(1, nbins):
        if omega[i] > 0 and omega[i] < 1:
            between_var[i] = (mu[nbins - 1] * omega[i] - mu[i]) ** 2 / (omega[i] * (1 - omega[i]))

    # Encontrar el umbral óptimo
    for i in range(1, nbins - 1):
        if between_var[i] > max_between_var:
            max_between_var = between_var[i]
            best_threshold = i

    # Aplicar el umbral a la imagen original
    _, binary_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)

    return binary_image


# Image Roi Select

def image_roi_select(image_binary, dicom_file):
    # Crear una figura para mostrar la imagen binarizada
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_binary, cmap='gray')

    binary_image_roi = []
    # Inicializar la selección de polígonos en la figura
    vertices = []
    polygon_selector = PolygonSelector(ax, onselect=None)

    # Conectar la función para aplicar la ROI después de seleccionarla
    polygon_selector.onselect = lambda vertices: apply_roi(image_binary, vertices, ax, binary_image_roi)

    plt.show()

    # Cuando la ROI está lista y deseas guardarla en el archivo DICOM original
    if binary_image_roi:
        binary_image_roi[0] = (binary_image_roi[0].astype(np.float32) / 255.0 * 65535.0 - 32768.0).astype(np.int16)
        dicom_file_with_roi = apply_roi_to_dicom(dicom_file, binary_image_roi[0])
        return dicom_file_with_roi
    else:
        print("No se ha seleccionado una región de interés (ROI).")
