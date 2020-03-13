import sys
import pytesseract
from pandas import ExcelFile
from PIL import Image
import os
import time
import csv
import pdf2image
import statistics
from PIL import ImageFile
from bounding_boxes import create_bounding_boxes
ImageFile.LOAD_TRUNCATED_IMAGES = True
#does this work
def pdf_to_img(pdf_file:str):
    return pdf2image.convert_from_path(pdf_file, dpi=300)

def typeFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def create_mapping(w2_dir_list: list, truth_file_name_list: list) -> dict:
    # maps w2 folder index to truth excel index
    mapping = {}
    for truth_file_index, truth_file_name in enumerate(truth_file_name_list):
        # noisy data set ends with .jpg
        if truth_file_name.endswith('.jpg'):
            if truth_file_name in w2_dir_list:
                w2_index = w2_dir_list.index(truth_file_name)
                mapping[w2_index] = truth_file_index
        else:
            # choose jpgs first
            if truth_file_name + '.jpg' in w2_dir_list:
                w2_index = w2_dir_list.index(truth_file_name + '.jpg')
                mapping[w2_index] = truth_file_index
            # choose pdf if jpgs do not exist
            elif truth_file_name + '.pdf' in w2_dir_list:
                w2_index = w2_dir_list.index(truth_file_name + '.pdf')
                mapping[w2_index] = truth_file_index
    return mapping

def evaluate(w2_folder:str, truth:str, sheet:int, starting_index:int, sample_type:str, results_csv:str) -> None:
    folder_list = [w2_folder]
    truth_list = [truth]
    #dir = '/Users/Taaha/Documents/projects'
    dir = 'data/fake-w2-us-tax-form-dataset' #'/Users/umaymahsultana/Desktop/data' 

    for folder_index, folder_dir in enumerate(folder_list):
        # set up paths for image folder and excel file
        folder_path = os.path.join(dir, folder_dir)
        excel_path = os.path.join(dir, truth_list[folder_index])

        # convert excel into readable pandas format and then convert to python dict
        xls = ExcelFile(excel_path)

        # have two dataframes, one that is used to get the correct file names and one for the truth
        file_names_df = xls.parse(xls.sheet_names[sheet])
        truth_df = xls.parse(xls.sheet_names[0])

        file_name_docs = file_names_df.to_dict('records')
        truth_docs = truth_df.to_dict('records')

        # get all the w2 image files in folder in a sorted manner
        files = sorted(os.listdir(folder_path))

        truth_file_name_list = []

        # get the file names from the specified sheet
        if sheet == 0:
            for doc in file_name_docs:
                truth_file_name_list.append(doc['File_BaseName'])
        elif sheet == 1:
            for doc in file_name_docs:
                truth_file_name_list.append(doc['file_name'])
        else:
            raise ValueError("Invalid sheet number")

        # maps w2 dir index to excel index in the truth set
        doc_mapping = create_mapping(files, truth_file_name_list)
        doc_items = doc_mapping.items()

        with open(results_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            # write the headers
            writer.writerow(['Document Name', 'Accuracy', 'Time (seconds)', 'Integer Accuracy', 'String Accuracy'])

            # save accuracy and time to compute averages
            accuracy_list = []
            time_list = []
            float_accuracy_list = []
            string_accuracy_list = []
            floatCorrect = 0
            floatWrong = 0
            stringCorrect = 0
            stringWrong = 0
            for w2_index, truth_index in doc_items:
                # get file in dir
                file = files[w2_index]

                # get truth set
                doc = truth_docs[truth_index]

                # Open text file with headers
                with open ("headers.txt", "r") as myFile : 
                	headers = myFile.read().split()

                full_file_path = os.path.join(folder_path, file)
                doc_name = 'W2_' + sample_type + '_' + str(truth_index + starting_index) + '_DataSet' + str(sheet) + file[-4:]
                # start timer
                start_time = time.time()
                # convert pdf to img
                if file.endswith('pdf'):
                    image = pdf_to_img(full_file_path)[0]
                else:
                    image = Image.open(full_file_path)
                parse = pytesseract.image_to_string(image)
                # end timer
                end_time = time.time()
                num_correct = 0
                num_total = 0
                field_names = doc.values()
                for field_name in field_names:
                    if str(field_name) in parse:
                        num_correct += 1
                        if(typeFloat(str(field_name))):
                            floatCorrect += 1

                        else:
                            stringCorrect += 1
                        parse.replace(str(field_name),'',1)
                    else:
                        if(typeFloat(str(field_name))):
                            floatWrong += 1
                        else:
                            stringWrong += 1
                    num_total += 1

                # code to check for headers
                for heading in headers:
                    if str(heading) in parse:
                        num_correct += 1
                        if (typeFloat(str(heading))):
                        	floatCorrect += 1

                        else: 
                        	stringCorrect += 1
                        parse.replace(str(heading), '', 1)

                    else: 
                    	if (typeFloat(str(heading))):
                    		floatWrong += 1
                    	else:
                    		stringWrong += 1
                    num_total += 1

                accuracy = (num_correct / num_total) * 100
                time_spent = end_time - start_time
                floatAccuracy = (floatCorrect / (floatCorrect+floatWrong)) * 100
                stringAccuracy = (stringCorrect / (stringCorrect + stringWrong)) * 100
                print(doc_name)
                print("Accuracy", accuracy)
                print("float Accuracy", floatAccuracy)
                print("String Accuracy", stringAccuracy)
                print("Time to parse document: {} seconds".format(time_spent))

                accuracy_list.append(accuracy)
                time_list.append(time_spent)
                float_accuracy_list.append(floatAccuracy)
                string_accuracy_list.append(stringAccuracy)

                writer.writerow([doc_name, accuracy, time_spent,floatAccuracy, stringAccuracy])

                # output images and text to a separate file
                #boxes_output_dir = "/Users/Taaha/Documents/projects"
                boxes_output_dir = "data/boxes/" #'/Users/umaymahsultana/Desktop/output' 
                if not os.path.exists(boxes_output_dir):
                    os.mkdir(boxes_output_dir)
                create_bounding_boxes(file, full_file_path, boxes_output_dir)

                #text_output_dir = "/Users/Taaha/Documents/projects"
                text_output_dir = "data/text/" #'/Users/umaymahsultana/Desktop/output' 
                if not os.path.exists(text_output_dir):
                    os.mkdir(text_output_dir)
                text_file = file.replace(".jpg", '.txt')
                with open(os.path.join(text_output_dir, text_file), 'w') as text_output:
                    text_output.writelines(parse)

            # write the averages on the last line
            accuracy_mean = statistics.mean(accuracy_list)
            time_mean = statistics.mean(time_list)
            float_accuracy_mean = statistics.mean(float_accuracy_list)
            string_accuracy_mean = statistics.mean(string_accuracy_list)
            writer.writerow(["Average", accuracy_mean, time_mean, float_accuracy_mean, string_accuracy_mean])

if __name__ == '__main__':
    evaluate('W2_Clean_DataSet_01_20Sep2019','W2_Truth_and_Noise_DataSet_01.xlsx', 0, 1000, 'Clean', 'W2_Clean_DataSet1_RESULTS.csv')
    evaluate('W2_Noise_DataSet_01_20Sep2019', 'W2_Truth_and_Noise_DataSet_01.xlsx', 1, 1000, 'Noisy','W2_Noisy_DataSet1_RESULTS.csv')
    evaluate('w2_samples_multi_clean', 'W2_Truth_and_Noise_DataSet_02.xlsx', 0, 5000,  'Clean', 'W2_Clean_DataSet2_RESULTS.csv')
    evaluate('w2_samples_multi_noisy', 'W2_Truth_and_Noise_DataSet_02.xlsx', 1, 5000,  'Noisy', 'W2_Noisy_DataSet2_RESULTS.csv')
