import sys
import pytesseract
from pandas import *
from PIL import Image
import os

if __name__ == '__main__':
    # CALL FUNCTION AS SUCH: $python evalulate.py [IMAGE DIRECTORY] [.XLXS DIRECTORY]
    exceldir = sys.argv[2]
    xls = ExcelFile(exceldir)
    df = xls.parse(xls.sheet_names[0])
    dict = df.to_dict()

    for index, key in enumerate(dict):
        if index == 1:
            print(dict[key])

    dir = sys.argv[1]
    files = sorted(os.listdir(dir))
    index = 0
    file_name_list = []
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            truncated_name = file[0:-4]
            file_name_list.append(file)
            file = os.path.join(dir, file)
            parse = pytesseract.image_to_string(Image.open(file))
            print (parse)
