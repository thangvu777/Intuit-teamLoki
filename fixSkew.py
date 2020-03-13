import pytesseract
from pytesseract import Output
import time
import cv2
import os
import numpy as np


def create_bounding_boxes(image_name:str, full_image_dir:str, output_dir:str) -> None:
    img = cv2.imread(full_image_dir)
    scale_percent = 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height),interpolation = cv2.INTER_LINEAR)
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if not cv2.imwrite(os.path.join(output_dir,image_name), img):
        raise Exception("Could not write image")


    #cv2.imsave(os.path.join(output_dir,image_name), img)
    #cv2.imshow(full_image_dir, img)
    #cv2.waitKey(0)

def fix_skew(img: str):
    skew_output = 'data/skew/fix_skew_output/'
    img = cv2.imread(img)
    # convert the image to grayscale and flip the foreground
    # background -> black; foreground -> white
    # background pixels = 0; foreground pixels = 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
	    angle = -angle
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Input", img)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)

def process ( directory : str , fix_skew : bool) -> None:
    for filename in sorted(os.listdir(directory)):
        start_time = time.time()
        if filename.endswith('.jpg'):
            file = os.path.join(directory, filename)
            print('OUTPUT of %s: ' %(file))
            print(pytesseract.image_to_string(file))
            save = filename[:-4] + '_boxes' + '.jpg'
            #create_bounding_boxes(save, file, 'truncated/out')
        else:
            continue
        end_time = time.time()
        total = end_time - start_time
        print('========================================================')
        print('%s took %.2f seconds to finish.' % (filename, total))
        print('========================================================')

if __name__ == '__main__':
    dir = 'data/skewed/W2_XL_input_noisy_1017.jpg'
    fix_skew(dir)
