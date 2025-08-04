import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
image_path = 'input_images/20701068_Q6_input.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# custom sharpening kernel
custom_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Padding
padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
rows, cols = image.shape

sharpened_image = np.zeros_like(image, dtype=np.float32)

# Convolution
for i in range(rows):
    for j in range(cols):
        region = padded_image[i:i+3, j:j+3]
        conv_value = np.sum(region * custom_kernel)
        sharpened_image[i, j] = conv_value

sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

# Display image
plt.figure(figsize=(10, 5))
plt.subplot(1, 1, 1)
plt.title("After applying custom sharpening filter")
plt.imshow(sharpened_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
