import cv2
import numpy as np
import matplotlib.pyplot as plt


# Apply Roi Mask
# Función para aplicar la máscara a la imagen binarizada
def apply_roi_mask(image, vertices):
    mask = np.zeros_like(image)
    vertices = np.array(vertices)  # Convierte a un array NumPy
    cv2.fillPoly(mask, [np.int32(vertices)], 1)
    return image * mask


# Apply Roi
# Función para aplicar la máscara después de seleccionar la región de interés
def apply_roi(binary_image, vertices, ax, binary_image_roi):
    if len(vertices) > 0:
        roi = apply_roi_mask(binary_image.copy(), vertices)  # Usar una copia de la imagen binarizada
        ax.clear()
        ax.imshow(roi, cmap='gray')
        plt.show()
        binary_image_roi[:] = [roi]


# Apply Roi To Dicom
def apply_roi_to_dicom(dicom, roi_data):
    if roi_data is not None:
        dicom_new = dicom.copy()
        dicom_new.PixelData = roi_data.tobytes()
        # dicom_new.Rows, dicom_new.Columns = dicom.shape
        return dicom_new
    else:
        return None
