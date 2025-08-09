import cv2
import matplotlib.pyplot as plt

img = cv2.imread('IMG_5468.jpeg')

'''
Image opens in a different color
Reason - MATPLOTLIB --> RGB (RED, GREEN, BLUE)
# OPENCV --> BGR (BLUE, GREEN, RED)
'''
plt.imshow(img)

'''
Convert the image color to RGB
'''
new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(new_img)

# Read image in GRAYSCALE
gray_image = cv2.imread('IMG_5468.jpeg', cv2.IMREAD_GRAYSCALE)
# It will not have any 3rd dimension because the image was read in GRAYSCALE. RGB channel isn't required
print(gray_image.shape)
plt.imshow(gray_image, cmap='gray')

'''
IMAGE THRESHOLDING
------------------
- Thresholding is fundamentally a very simple method of segmenting an image into different parts
- Thresholding will convert an image to consist of only two values, white or black (binary)

Types
1. Binary Threshold
2. Binary Inverse Threshold
3. Truncate Threshold
4. To Zero Threshold
5. To Zero Inverse Threshold
'''
r, threshold_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
plt.imshow(threshold_img, cmap='gray')

'''
Adaptive Thresholding
- The threshold value is calculated separately for each pixel based on the pixel’s neighborhood.
- This makes it adapt to different lighting conditions within the same image.
- Two methods in OpenCV:
    - cv2.ADAPTIVE_THRESH_MEAN_C → Uses mean of neighboring pixels.
    - cv2.ADAPTIVE_THRESH_GAUSSIAN_C → Uses weighted sum (Gaussian) of neighbors.
    
Works well with uneven lighting.
Better detail preservation in varying contrast areas.
'''
adaptive_threshold_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
plt.imshow(adaptive_threshold_img)

'''
BLURRING and SMOOTHING
----------------------
Blurring or smoothing is combined with edge detection
Blurring and smoothing are image processing techniques used to reduce noise and detail in an image,
making it look “softer” or less sharp.
Why we use them
1. Remove noise (random variations in brightness or color)
2. Reduce detail before other processing (like edge detection or segmentation)
3. Create artistic effects (e.g., motion blur, depth of field)
4. Preprocess images for object detection

**** Various methods ****

1. Gamma Correction
It can be applied to an image to make it appear brighter or darker depending on the Gamma value chosen
gamma value : < 1 then image will appear brighter

2. Kernel Based Filters
A kernel (or convolution matrix) is a small matrix (e.g., 
used to process an image by sliding over it and applying a mathematical operation (convolution).
This is the foundation for blurring, sharpening, edge detection, etc.

General Convolution Process
a. Place the kernel over a pixel’s neighborhood.
b. Multiply each pixel value by the corresponding kernel value.
c. Sum them up and replace the center pixel.

'''


plt.show()
