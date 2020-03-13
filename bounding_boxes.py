import pytesseract
from pytesseract import Output
import cv2
import os
from numpy import asarray
def create_bounding_boxes(img, image_name:str, output_dir:str) -> None:
   #img = cv2.imread(full_image_dir)
    # convert image to numpy
    img = asarray(img)
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
    # replace pdf with jpg in order to write image
    if image_name.endswith('.pdf'):
        image_name = image_name.replace('.pdf', '.jpg')
    cv2.imwrite(os.path.join(output_dir,image_name), img)
