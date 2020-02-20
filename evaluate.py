import sys
import pytesseract
from pandas import ExcelFile
from PIL import Image
import os
import time
import csv
import pdf2image

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file, dpi=300)


if __name__ == '__main__':
    folder_list = ["W2_Clean_DataSet_01_20Sep2019", "W2_Noise_DataSet_01_20Sep2019"]
    truth_list = ['W2_Truth_and_Noise_DataSet_01.xlsx', "W2_Truth_and_Noise_DataSet_02.xlsx"]
    dir = 'data/fake-w2-us-tax-form-dataset'
    for folder_index, folder_dir in enumerate(folder_list):
        # set up paths for image folder and excel file
        folder_path = os.path.join(dir, folder_dir)
        excel_path = os.path.join(dir, truth_list[folder_index])

        # convert excel into readable pandas format
        xls = ExcelFile(excel_path)
        df = xls.parse(xls.sheet_names[0])
        docs = df.to_dict('records')

        files = sorted(os.listdir(folder_path))
        index = 0
        document_index = 1000
        with open("results.csv", 'w') as csv_file:
            writer = csv.writer(csv_file)
            # write the headers
            writer.writerow(['Document Name', 'Accuracy', 'Time (seconds)'])
            for file in files:
                # select only one image type per document
                if str(document_index) in file:
                    file = os.path.join(folder_path, file)
                    doc = docs[index]
                    doc_name = doc['File_BaseName']
                    start_time = time.time()
                    # convert pdf to img
                    if file.endswith('pdf'):
                        image = pdf_to_img(file)[0]
                    else:
                        image = Image.open(file)
                    parse = pytesseract.image_to_string(image)
                    end_time = time.time()
                    num_correct = 0
                    num_total = 0
                    field_names = doc.values()
                    for field_name in field_names:
                        if str(field_name) in parse:
                            num_correct += 1
                        num_total += 1
                    accuracy = (num_correct / num_total) * 100
                    time_spent = end_time - start_time
                    print(doc_name)
                    print("Accuracy", accuracy)
                    print("Time to parse document: {} seconds".format(time_spent))
                    writer.writerow([doc_name, accuracy, time_spent])
                    index += 1
                    document_index += 1

