import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input Image
image_path = 'input_images/20701068_Q3_input.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Laplacian Sharpening filter (3*3)
sharpening_kernel = np.array([
    [0, 1,  0],
    [1, -4, 1],
    [0, 1,  0]
], dtype=np.float32)


sharpened_image = cv2.filter2D(src=image, ddepth=-1, kernel=sharpening_kernel)

# Display Image
plt.subplot(1, 1, 1)
plt.title("After applying Sharpening Filter")
plt.imshow(sharpened_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
