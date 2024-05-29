import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob
from functions import *
from tkinter import filedialog

# Load the image

image_path = 'C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/5-29-24 Biref in constriction direction/MB03x5 ph4.5/10nlmin 0deg.tif'
image_0_path = 'C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/5-29-24 Biref in constriction direction/MB03x5 ph4.5/img0 0.5ms exp.tif'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load the image in unchanged mode
img0 = cv2.imread(image_0_path, cv2.IMREAD_UNCHANGED) 
# # Display the original image
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.show()


# Display the original images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('First Image')
plt.subplot(1, 2, 2)
plt.imshow(img0, cmap='gray')
plt.title('Second Image')
plt.show()

# Avoid division by zero by replacing zeros in image2 with a small value
img0_safe = np.where(img0 == 0, 1, img0)

# Perform the pixel-wise division
divided_image = np.divide(image, img0_safe)

# Normalize the result to the 16-bit range (0-65535)
# Display the resulting image
plt.imshow(divided_image, cmap='gray')
plt.title('Divided Image')
plt.show()



# Contrast Stretching
min_val = np.min(image)
print(min_val)
max_val = np.max(image)
print(max_val)
stretched_image = ((image - 9000) / (max_val - 9000) * 256).astype(np.uint8)

# Display the contrast stretched image
plt.imshow(stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')
plt.show()