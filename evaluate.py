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


def evaluate(w2_folder, truth, sheet, starting_index, results_csv): #(sheet(int): 1 for truth sheet 1, 2 for truth sheet 2)
    folder_list = [w2_folder]
    truth_list = [truth]
    dir = 'data/fake-w2-us-tax-form-dataset'

    idx = 0 #Dataset one files are (1000-2999) #Dataset two files are (10499-15500)
    if sheet == 1:
        idx = -4
    if sheet == 2:
        idx = -5

    for folder_index, folder_dir in enumerate(folder_list):
        # set up paths for image folder and excel file
        folder_path = os.path.join(dir, folder_dir)
        excel_path = os.path.join(dir, truth_list[folder_index])

        # convert excel into readable pandas format
        xls = ExcelFile(excel_path)
        df = xls.parse(xls.sheet_names[0])

        # truth list members: used to verify that the file is there, otherwise skip
        # Indexing
        names = []
        check = []
        names.append(df['File_BaseName'])
        for i in range (0, len(names[0])):
            check.append((names[0][i])[idx:]) #File 1000 is in there, 1001, etc...

        docs = df.to_dict('records')
        files = sorted(os.listdir(folder_path))
        index = 0
        document_index = starting_index

        with open(results_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            # write the headers
            writer.writerow(['Document Name', 'Accuracy', 'Time (seconds)'])
            for file in files:
                #check to see if file is in the directory, if not skip to next iteration
                stripped = (file[:-4])[idx:]
                if (stripped in check):
                    # select only one image type per document
                    if str(document_index) in file:
                        file = os.path.join(folder_path, file)
                        if not os.path.isfile(file): 
                            index += 1
                            document_index += 1
                            continue
                        else:
                            doc = docs[index]
                            doc_name = doc['File_BaseName']
                            start_time = time.time()
                            # convert pdf to img
                            if file.endswith('pdf'):
                                image = pdf_to_img(file)[0]
                            else:
                                image = Image.open(file)
                            try:
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
                            except: #Truncated images won't scan with tesseract so skip the file
                                print("=========================================================================================")
                                print("Exception raised when processing image:")
                                print (file)
                                print("=========================================================================================")
                                index += 1
                                document_index += 1
                                continue
                else:
                    index += 1
                    document_index += 1
                    continue




if __name__ == '__main__':
    # evaluate('W2_DIRECTORY', 'W2_TRUTH_FILE', DATASET# (1 OR 2), STARTING INDEX (1000 OR 10499), 'CSV OUTPUT NAME')
    evaluate('W2_Clean_DataSet_01_20Sep2019','W2_Truth_and_Noise_DataSet_01.xlsx', 1, 1000, 'W2_Clean_DataSet1_RESULTS')
    evaluate('W2_Noise_DataSet_01_20Sep2019', 'W2_Truth_and_Noise_DataSet_01.xlsx', 1, 1000, 'W2_Noisy_DataSet1_RESULTS')
    evaluate('w2_samples_multi_clean', 'W2_Truth_and_Noise_DataSet_02.xlsx', 2, 10499, 'W2_Clean_DataSet2_RESULTS' )
    evaluate('w2_samples_multi_noisy', 'W2_Truth_and_Noise_DataSet_02.xlsx', 2. 10499, 'W2_Noisy_DataSet2_RESULTS')
