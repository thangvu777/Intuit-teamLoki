# shadow removal - only using a single jpeg at a time

import cv2
import os
import numpy as np

# clean image
# image_path = '/Users/brianlai/Desktop/preprocessing_Test/W2_XL_input_clean_1103.jpg'
# noisy image
image_path = '/Users/brianlai/Desktop/preprocessing_Test/W2_XL_input_real_noisy_1103.jpeg'
dir = '/Users/brianlai/Desktop/preprocessing_Test'

os.chdir(dir)

print("Before saving image: ")
print(os.listdir(dir))

image = cv2.imread(image_path, -1)

rgb_planes = cv2.split(image)

result_planes = []
result_norm_planes = []

for plane in rgb_planes:
    # dilate image to remove text
    dilated_image = cv2.dilate(plane, np.ones((7, 7), np.uint8))

    # suppress any text with median blur to get background with all the shadows/discoloration
    bg_image = cv2.medianBlur(dilated_image, 21)

    # invert the result by calculating difference between original and bg_image, looking for black on white
    diff_image = 255 - cv2.absdiff(plane, bg_image)

    # normalize image to use full dynamic range
    norm_image = cv2.normalize(diff_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    result_planes.append(diff_image)
    result_norm_planes.append(norm_image)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

cv2.imwrite('W2_XL_input_real_noisy_1103_shadow_out.jpg', result)
cv2.imwrite('W2_XL_input_real_noisy_1103_shadow_out_norm.jpg', result_norm)

print("After saving image: ")
print(os.listdir(dir))
print('Successfully saved -Shadow Removal- result')
