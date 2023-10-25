import numpy as np
import cv2

def convolution(image, kernel):
    # Obtém as dimensões da imagem e do kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calcula o padding para manter o tamanho da imagem de saída igual ao original
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Cria uma imagem de saída preenchida com zeros
    output = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # Aplica a convolução
    for i in range(pad_height, image_height - pad_height):
        for j in range(pad_width, image_width - pad_width):
            # Realiza a operação de convolução
            output[i, j] = np.sum(image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1] * kernel)
    
    return output

image = cv2.imread('teste.jpg', cv2.IMREAD_GRAYSCALE)

# Define um kernel de exemplo (filtro de borda)
kernel = np.array([[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]])

# Realiza a convolução
result = convolution(image, kernel)

# Salva a imagem resultante
cv2.imwrite('convolucao.jpg', result)
