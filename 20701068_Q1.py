import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image
image_path = 'input_images/20701068_Q1_input.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 3x3 averaging kernel
kernel = (1/9) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32)
rows, cols = image.shape
padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
smoothed_image = np.zeros_like(image, dtype=np.uint8)

# convolution 
for i in range(rows):
    for j in range(cols):
        region = padded_image[i:i+3, j:j+3]
        smoothed_value = np.sum(region * kernel)
        smoothed_image[i, j] = np.clip(smoothed_value, 0, 255)

# Display image
plt.figure(figsize=(10, 4))

plt.subplot(1, 1, 1)
plt.title("After applying Averaging Filter")
plt.imshow(smoothed_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
