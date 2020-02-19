import sys
import pytesseract
from pandas import ExcelFile
from PIL import Image
import os
import time
import csv

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
        with open("results.csv", 'w') as csv_file:
            writer = csv.writer(csv_file)
            # write the headers
            writer.writerow(['Document Name', 'Accuracy', 'Time (seconds)'])
            for file in files:
                # filter based on images
                if file.endswith('.png') or file.endswith('.jpg'):
                    file = os.path.join(folder_path, file)
                    doc = docs[index]
                    doc_name = doc['File_BaseName']
                    start_time = time.time()
                    parse = pytesseract.image_to_string(Image.open(file))
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


