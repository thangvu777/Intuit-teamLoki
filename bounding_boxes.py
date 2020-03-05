import pytesseract
from pytesseract import Output
import cv2
import os

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
    cv2.imwrite(os.path.join(output_dir,image_name), img)
