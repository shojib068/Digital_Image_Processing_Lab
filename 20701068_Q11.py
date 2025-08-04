import cv2
import numpy as np
import matplotlib.pyplot as plt

def ideal_band_reject_filter(shape, r1, r2):

    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)

    mask = np.ones((rows, cols), dtype=np.float32)
    mask[(D >= r1) & (D <= r2)] = 0
    return mask

# Input Image
img = cv2.imread('input_images/20701068_Q11_input.tif', 0)

# DFT and shift
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Define band to reject
r1, r2 = 30, 80

# Band-reject filter mask
mask = ideal_band_reject_filter(img.shape, r1, r2)
mask_2ch = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
fshift = dft_shift * mask_2ch

# Inverse DFT
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)

# Display Image
plt.figure(figsize=(10, 5))

plt.subplot(1, 1, 1)
plt.imshow(img_back, cmap='gray')
plt.title(f'Ideal Band-Reject Filter\nBand: {r1} to {r2}')
plt.axis('off')

plt.tight_layout()
plt.show()
