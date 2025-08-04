import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
img = cv2.imread('input_images/20701068_Q7_input.tif', cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)


fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Display Output
plt.figure(figsize=(15, 5))

plt.subplot(1, 1, 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT Magnitude Spectrum')
plt.axis('off')

plt.tight_layout()
plt.show()
