import cv2
import numpy as np
import matplotlib.pyplot as plt
# Input Image
bgr_img = cv2.imread('input_images/20701068_Q13_input.jpeg')
# BGR to RGB
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

#  RGB to HSV
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(hsv_img)

# Display Images
plt.figure(figsize=(15, 6))

plt.subplot(1, 4, 1)
plt.title('Orginal')
plt.imshow(rgb_img)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Hue(H)')
plt.imshow(H, cmap='hsv')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Saturation(S)')
plt.imshow(S, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Value(V)')
plt.imshow(V, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
