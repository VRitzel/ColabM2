import numpy as np
import cv2
import filtros as f

# Fiz esse arquivo pois eram muitas imagens

# Aplica a filtragem de suavização com filtro de média, gaussiano e mediana
smoothed_image_3x3_media = f.apply_smoothing(f.image, 'media', 3)
smoothed_image_3x3_gaussiano = f.apply_smoothing(f.image, 'gaussiano', 3)
smoothed_image_3x3_mediana = f.apply_smoothing(f.image, 'mediana', 3)

smoothed_image_5x5_media = f.apply_smoothing(f.image, 'media', 5)
smoothed_image_5x5_gaussiano = f.apply_smoothing(f.image, 'gaussiano', 5)
smoothed_image_5x5_mediana = f.apply_smoothing(f.image, 'mediana', 5)

smoothed_image_7x7_media = f.apply_smoothing(f.image, 'media', 7)
smoothed_image_7x7_gaussiano = f.apply_smoothing(f.image, 'gaussiano', 7)
smoothed_image_7x7_mediana = f.apply_smoothing(f.image, 'mediana', 7)

# Salva as imagens resultantes
cv2.imwrite('smoothed_3x3_media.jpg', smoothed_image_3x3_media)
cv2.imwrite('smoothed_3x3_gaussiano.jpg', smoothed_image_3x3_gaussiano)
cv2.imwrite('smoothed_3x3_mediana.jpg', smoothed_image_3x3_mediana)
cv2.imwrite('smoothed_5x5_media.jpg', smoothed_image_5x5_media)
cv2.imwrite('smoothed_5x5_gaussiano.jpg', smoothed_image_5x5_gaussiano)
cv2.imwrite('smoothed_5x5_mediana.jpg', smoothed_image_5x5_mediana)
cv2.imwrite('smoothed_7x7_media.jpg', smoothed_image_7x7_media)
cv2.imwrite('smoothed_7x7_gaussiano.jpg', smoothed_image_7x7_gaussiano)
cv2.imwrite('smoothed_7x7_mediana.jpg', smoothed_image_7x7_mediana)