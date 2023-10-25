import numpy as np
import cv2
import main as m

def apply_filter(image, kernel):
    return m.convolution(image, kernel)

def apply_smoothing(image, filter_type, kernel_size):
    if filter_type == 'media':
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    elif filter_type == 'gaussiano':
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16
    elif filter_type == 'mediana':
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError("Tipo de filtro não suportado.")
    
    return apply_filter(image, kernel)

image = cv2.imread('teste.jpg', cv2.IMREAD_GRAYSCALE)


