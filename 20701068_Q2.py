import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
image_path = 'input_images/20701068_Q2_input.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 3x3 Gaussian kernel
gaussian_k = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

gaussian_blurred_manual = cv2.filter2D(image, -1, gaussian_k)

# Display image
plt.subplot(1, 1, 1)
plt.title("After applying Gaussian Filter")
plt.imshow(gaussian_blurred_manual, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
