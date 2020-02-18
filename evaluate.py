import sys
import pytesseract
from PIL import Image
import os
import csv
if __name__ == '__main__':
    dir = sys.argv[1]
    files = os.listdir(dir)
    for file in files:
        if file.endswith('.png') or file.endswith('jpg') or file.endswith('.pdf'):
            file = os.path.join(dir, file)
            parse = pytesseract.image_to_data(Image.open(file))
