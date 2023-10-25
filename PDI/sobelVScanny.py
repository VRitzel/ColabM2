import numpy as np
import cv2
import filtros
import main

#smoothed_image = filtros.smoothed_image_3x3_mediana # Não preciso reimportar dessa vez :)
smoothed_image = filtros.apply_smoothing(filtros.image, 'media', 3)
# Deja vu
kernel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)
kernel_y = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=np.float32)

gradient_x = main.convolution(smoothed_image, kernel_x)
gradient_y = main.convolution(smoothed_image, kernel_y)

# Aplica ambos sobel e escreve a imagem
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
cv2.imwrite('sobel_mediana_resultado.jpg', magnitude)

# Aplica e salva o canny edge
canny_edge = cv2.Canny(smoothed_image, 100, 200)
cv2.imwrite('canny_edge_mediana.jpg', canny_edge)
