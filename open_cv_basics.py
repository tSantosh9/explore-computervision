import cv2
import matplotlib.pyplot as plt

img = cv2.imread('IMG_5468.jpeg')

# Image opens in a different color
# Reason - MATPLOTLIB --> RGB (RED, GREEN, BLUE)
# OPENCV --> BGR (BLUE, GREEN, RED)
#plt.imshow(img)

# Convert the image color to RGB
#new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(new_img)

# Read image in GRAYSCALE
gray_image = cv2.imread('IMG_5468.jpeg', cv2.IMREAD_GRAYSCALE)
# It will not have any 3rd dimension because the image was read in GRAYSCALE. RGB channel isn't required
print(gray_image.shape)
plt.imshow(gray_image, cmap='gray')

plt.show()
