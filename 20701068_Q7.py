import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
image_path = 'input_images/cameraman.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Compute FFT and shift
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_shift) + 1)

# Extract 1D slice from the center row
center_row = magnitude_spectrum.shape[0] // 2
line_data = magnitude_spectrum[center_row, :]

# Display the graph
plt.figure(figsize=(10, 5))
plt.plot(line_data, color='blue')
plt.title("1D Visualization of FFT")
plt.xlabel("Frequency Index")
plt.ylabel("Log Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()
