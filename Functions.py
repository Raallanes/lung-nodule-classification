import cv2
import numpy as np
import matplotlib.pyplot as plt

def kurita_binarization(Image,nbins,k):
    Image = Image.astype(float)

    # Calculate histogram
    h, cbin = np.histogram(Image, bins=nbins, density=True)
    h = h / np.sum(h)

    # Find the range of significant values
    max_indx = np.argmax(h)
    min_indx = np.argmin(h)

    # Calculate total mean and total variance
    totalMean = np.mean(h[min_indx:max_indx + 1])
    totalvar = np.var(h[min_indx:max_indx + 1])

    prevProb1 = 0
    mean1 = 0
    cf = np.zeros(nbins)

    for i in range(min_indx, max_indx + 1):
        prob1 = prevProb1 + h[i]
        prob2 = 1 - prob1
        mean1 = (prevProb1 * mean1 + h[i] * i) / prob1
        mean2 = np.mean(h[i + 1:max_indx + 1])
        t1 = mean1 - mean2
        cf[i] = np.log(totalvar - prob1 * prob2 * t1 * t1) - prob1 * np.log(prob1) + prob2 * np.log(prob2)
        prevProb1 = prob1

    cf = cf / totalvar
    thindx = np.argmin(cf[:-1])
    thrKur = cbin[thindx] * k
    bwKur = (Image > thrKur).astype(np.uint8) * 255

    # Adjust threshold value based on the original data type
    if Image.dtype == np.uint8:
        thrKur = np.round(255 * thrKur)
    elif Image.dtype == np.uint16:
        thrKur = np.round(65535 * thrKur)

    return bwKur, thrKur

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

# Función para aplicar la máscara a la imagen binarizada
def apply_roi_mask(image, vertices):
    mask = np.zeros_like(image)
    vertices = np.array(vertices)  # Convierte a un array NumPy
    cv2.fillPoly(mask, [np.int32(vertices)], 1)
    return image * mask

# Función para aplicar la máscara después de seleccionar la región de interés
def apply_roi(binary_image, vertices, ax, binary_image_roi):
    if len(vertices) > 0:
        roi = apply_roi_mask(binary_image.copy(), vertices)  # Usar una copia de la imagen binarizada
        ax.clear()
        ax.imshow(roi, cmap='gray')
        plt.show()
        binary_image_roi[:] = [roi]

def apply_roi_to_dicom(dicom, roi_data):
    if roi_data is not None:
        dicom_new = dicom.copy()
        dicom_new.PixelData = roi_data.tobytes()
        #dicom_new.Rows, dicom_new.Columns = dicom.shape
        return dicom_new
    else:
        return None

def show_image(text,img):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


