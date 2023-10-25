import cv2
import numpy as np
import matplotlib.pyplot as plt
import main as m

# Carregar a imagem
image = cv2.imread('teste.jpg', cv2.IMREAD_GRAYSCALE)

laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)
laplacian = cv2.filter2D(image, -1, laplacian_kernel)
#laplacian = m.convolution(image, laplacian_kernel)
cv2.imwrite("Laplacian Filter.jpg", laplacian)

sharpening_kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]], dtype=np.float32)
sharpened = cv2.filter2D(image, -1, sharpening_kernel)
cv2.imwrite("Sharpening Filter.jpg", sharpened)

# Aplicar um filtro gaussiano para obter a imagem desfocada
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Calcular a máscara de realce
unsharp_mask = image - blurred
highboost = image + 1.5 * unsharp_mask  # Ajuste o fator para controlar o realce
cv2.imwrite("Unsharp Masking.jpg", unsharp_mask)
cv2.imwrite("Highboost Filtering.jpg", highboost)

laplacian_edge = cv2.Laplacian(image, cv2.CV_64F)
roberts_x = cv2.filter2D(image, -1, np.array([[1, 0], [0, -1]], dtype=np.float32))
roberts_y = cv2.filter2D(image, -1, np.array([[0, 1], [-1, 0]], dtype=np.float32))

sobel_x_manual = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

sobel_y_manual = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)

sobel_x = cv2.filter2D(image, -1, sobel_x_manual)
sobel_y = cv2.filter2D(image, -1, sobel_y_manual)

# Use a função Canny da OpenCV para detecção de borda
canny_edge = cv2.Canny(image, 100, 200)

cv2.imwrite("Laplacian Edge.jpg", laplacian_edge)
cv2.imwrite("Roberts X.jpg", roberts_x)
cv2.imwrite("Roberts Y.jpg", roberts_y)
cv2.imwrite("Sobel X.jpg", sobel_x)
cv2.imwrite("Sobel Y.jpg", sobel_y)
cv2.imwrite("Canny Edge.jpg", canny_edge)
