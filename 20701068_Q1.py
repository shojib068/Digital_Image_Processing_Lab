import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
image_path = 'input_images/cameraman.tif'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 3*3 kernel
kernel = np.ones((3, 3), dtype=np.float32) / 9.0
rows, cols = image.shape
padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
smoothed_image = np.zeros_like(image)

# convolution
for i in range(rows):
    for j in range(cols):
        region = padded_image[i:i+3, j:j+3]
        smoothed_value = np.sum(region * kernel)
        smoothed_image[i, j] = np.clip(smoothed_value, 0, 255)
        
# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("After applying Averaging filter")
plt.imshow(smoothed_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
