import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input Image
img = cv2.imread('input_images/20701068_Q19_input.tif', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Connected components analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# remove objects < 150 pixels)
min_size = 150
cleaned = np.zeros_like(binary)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_size:
        cleaned[labels == i] = 255
        
# Display Image
plt.figure(figsize=(10,5))

plt.subplot(1,1,1)
plt.imshow(cleaned, cmap='gray')
plt.title(f'Removed Objects < {min_size} Pixels')
plt.axis('off')

plt.tight_layout()
plt.show()
