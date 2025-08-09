import cv2
import numpy as np

# ---- Load image ----
img = cv2.imread("IMG_5468.jpeg")  # Change to your file path
if img is None:
    raise FileNotFoundError("Image not found! Check the path.")

# ---- 1. Gamma Correction ----
gamma = 0.5  # <1 brightens, >1 darkens
gamma_corrected = np.power(img / 255.0, gamma)
gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

# ---- 2. Kernel-based Blur ----
kernel_blur = np.ones((5, 5), np.float32) / 25  # Average blur kernel
blurred = cv2.filter2D(img, -1, kernel_blur)

# ---- 3. Kernel-based Sharpen ----
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened = cv2.filter2D(img, -1, kernel_sharpen)

# ---- Combine all images horizontally ----
# Resize all to same height for display
height = 300
def resize(img):
    return cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))

combined = np.hstack([
    resize(img),
    resize(gamma_corrected),
    resize(blurred),
    resize(sharpened)
])

# ---- Show result ----
cv2.imshow("Original | Gamma Corrected | Blurred | Sharpened", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
