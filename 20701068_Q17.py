import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input Image
img = cv2.imread('input_images/cameraman.tif', cv2.IMREAD_GRAYSCALE)

# Binarize the image
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)

# Morphological opening
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Display Images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(opened, cmap='gray')
plt.title('After Opening (Small White Noise Removed)')
plt.axis('off')

plt.tight_layout()
plt.show()
