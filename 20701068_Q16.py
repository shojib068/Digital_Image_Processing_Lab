import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input Image
img = cv2.imread('input_images/20701068_Q16_input.tif', cv2.IMREAD_GRAYSCALE)

# Binarize image
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)

# Apply erosion
eroded = cv2.erode(binary, kernel, iterations=1)

# Display image
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(eroded, cmap='gray')
plt.title('Eroded Image')
plt.axis('off')

plt.tight_layout()
plt.show()
