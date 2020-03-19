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
from fuzzywuzzy import process, fuzz
import random
import cv2
import numpy as np
import math
#does this work
def pdf_to_img(pdf_file:str):
    return pdf2image.convert_from_path(pdf_file, dpi=300)


def typeFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def random_sample(truth_file_name_list:list, truth_docs:list, sample_size:int):
    # need to call the seed before using random to generate the same randon numbers
    random.seed(7)
    sample_file_names = random.sample(truth_file_name_list, sample_size)
    random.seed(7)
    sample_truth_docs = random.sample(truth_docs, sample_size)
    return sample_file_names, sample_truth_docs



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

def get_excel_docs(excel_path:str, sheet:int):
    # convert excel into readable pandas format and then convert to python dict
    xls = ExcelFile(excel_path)

    # have two dataframes, one that is used to get the correct file names and one for the truth
    file_names_df = xls.parse(xls.sheet_names[sheet])
    truth_df = xls.parse(xls.sheet_names[0])

    file_name_docs = file_names_df.to_dict('records')
    truth_docs = truth_df.to_dict('records')

    # get the file names from the specified sheet
    truth_file_name_list = []
    if sheet == 0:
        for doc in file_name_docs:
            truth_file_name_list.append(doc['File_BaseName'])
    elif sheet == 1:
        for doc in file_name_docs:
            truth_file_name_list.append(doc['file_name'])
    else:
        raise ValueError("Invalid sheet number")
    return truth_file_name_list, truth_docs

def fuzzy_evaluate(field_name:str, parse:str) -> None:
    # TODO: some field names are nan
    # set field names to a blank string if nan
    # if field_name == "nan":
    #     field_name = ""
    extract = process.extractOne(str(field_name), parse)
    # return the score
    return extract[1]

# Call various pre processing functions
def pre_process(w2Image):
	# w2Image = remove_shadow(w2Image)
	return w2Image


# Removes shadow and normalizes
def remove_shadow(imageIn):
    #image = cv2.imread(image_path, -1)
    image = np.array(imageIn)

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

    return result_norm

# Fix text skew of an image
def fix_skew(img: str):
    fix_skew_helper(img)

''' 
    Fix skew helper:
    1. Find the biggest contour outline of a given image
    2. Get the dimensions of the image using shape
    3. Find the approximate Cartesian coordinates of the image  
    4. Calculate the angle between 3 points (2 points from the image, 1 from (0,0))
    5. Check for straight image, if image is crooked, rotation will occur
    6. Rotate and crop the image using the center of the original image
'''
def fix_skew_helper(img: str):
    # save path string
    path = img

    # load image
    img = cv2.imread(img)

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold image
    thresh = cv2.threshold(gray, 4, 255, 0)[1]

    # apply morphology open to smooth the outline
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # find all contours
    cntrs = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # STEP 1: find biggest contour to outline the W2 form
    area_thresh = 0
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c

    '''
    # Uncomment if you want to see the contour lines
    # (RED) draw the contour on a copy of the input image
    results = img.copy()
    cv2.drawContours(results, [big_contour], 0, (0, 0, 255), 2)
    out = str('Contouring     ' + path)
    cv2.imshow(out, results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # STEP 2: get dimensions of image
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    #print ("w: %d h: %d" %(width, height))

    center = (width / 2, height / 2)

    # STEP 3: Find approximate coordinates of outlining box
        # .05 approx level gives us the 4 coordinates of the W-2 Rectangular shape
    approx = cv2.approxPolyDP(big_contour, .05 * cv2.arcLength(big_contour, True), True)
    # Used to flatten the array containing the co-ordinates of the vertices.
    n = approx.ravel()
    i = 0
    points = []
    for j in n:
        if (i % 2 == 0):
            x = n[i]
            y = n[i + 1]
        i = i + 1
        points.append([x,y])
        #print(x,y)

    # STEP 4: Calculate degree of rotation
    # FUNCTION DEFINITION: Calculate an angle given 3 Cartesian coordinates
    def getAngle(a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang

    '''
    This is for images that don't necessarily need to be rotated since they already stand correctly
         .00166 is equal to 1/600
         ex. Original Dimensions (w x h): 2005 x 2340
         1/600 * (2005) = 3.33; 1/600 * (2340) = 3.88
         If one of the vertices of the image lies at approximately 2004 x 2339
         the image is most likely straight because the absolute difference is 1 pixel apart 
    '''

    if (abs(width - points[4][0]) <= .00166 * width and abs(height - points[4][1] <= .00166 * height)):
        angle = 0
    else: # Rotation needs to occur
        # This is for images rotated counter-clockwise originally with respect to the center
        # X1 > Y2: ex. P1 = (1221,0); P2 = (0,172) -> True
        if (points[0][0] >= points[2][1]):
            angle = (getAngle([0, 0], points[0], points[2]))

            # adjust width and height for cropping
            width = int(abs(points[6][0] - points[4][0]))
            height = int(abs(points[6][1] - points[0][1]))
        # This is for images rotated clockwise originally with respect to the center
        # X1 < Y1: ex. P1 = (405,0); P2 = (0,2303) -> True
        elif (points[0][0] < points[2][1]):
            angle = (getAngle([0, 0], points[0], points[2])) + 90

            # adjust width and height for cropping
            width = int(abs(points[6][0] - points[0][0]))
            height = int(abs(points[4][1] - points[6][1]))

    #print('angle = %.3f' % angle)

    # STEP 5: Rotate and Crop
    # FUNCTION DEFINITION: Rotate a given image around the center with angle theta in degrees,
    # then the image is cropped according to width and height.
    def rotate(image, center: int, theta: float, width: int, height: int):
        # Uncomment to use theta instead
        # theta *= 180/np.pi

        shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

        matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
        image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)

        image = image[y:y + height, x:x + width]
        return image

    #print('w: %d, h: %d, center: %s ' %(width,height,center))

    rotated_image = rotate(img, center=center, theta=angle, width= width, height=height)

    '''
    # Uncomment if you want to see the rotated image
    out = str('Rotated     ' + path)
    cv2.imshow(out, rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # Return the final rotated image
    return rotated_image

def evaluate(w2_folder:str, truth:str, sheet:int, starting_index:int, sample_type:str, results_csv:str) -> None:
    folder_list = [w2_folder]
    truth_list = [truth]
    #dir = '/Users/Taaha/Documents/projects'
    dir = 'data/fake-w2-us-tax-form-dataset'
    #dir = '/Users/umaymahsultana/Desktop/data'

    for folder_index, folder_dir in enumerate(folder_list):
        # set up paths for image folder and excel file
        folder_path = os.path.join(dir, folder_dir)
        excel_path = os.path.join(dir, truth_list[folder_index])

        # returns the file names and the truth data of all images for that folder
        truth_file_name_list, truth_docs = get_excel_docs(excel_path, sheet)
        # randomly take 100 files
        sample_truth_file_list, truth_docs = random_sample(truth_file_name_list, truth_docs, 5)
        # get all the w2 image files in folder in a sorted manner
        files = sorted(os.listdir(folder_path))

        # maps w2 dir index to excel index in the truth set
        doc_mapping = create_mapping(files, sample_truth_file_list)
        doc_items = doc_mapping.items()

        with open(results_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            # write the headers
            writer.writerow(['Document Name', 'Accuracy', 'Time (seconds)', 'Integer Accuracy', 'String Accuracy', 'Fuzzy Score'])

            # save accuracy and time to compute averages
            accuracy_list = []
            time_list = []
            float_accuracy_list = []
            string_accuracy_list = []
            fuzzy_score_list = []

            floatCorrect = 0
            floatWrong = 0
            stringCorrect = 0
            stringWrong = 0
            for w2_index, truth_index in doc_items:

                # get file in dir
                file = files[w2_index]
                doc_name = file
                # get truth set
                doc = truth_docs[truth_index]

                # Open text file with headers
                with open ("headers.txt", "r") as myFile : 
                    headers = myFile.read().split()

                full_file_path = os.path.join(folder_path, file)
                # start timer
                start_time = time.time()
                # convert pdf to img
                if file.endswith('pdf'):
                    image = pdf_to_img(full_file_path)[0]
                else:
                    image = Image.open(full_file_path)
                image = pre_process(image) #calls pre process functions
                parse = pytesseract.image_to_string(image)
                # end timer
                end_time = time.time()
                num_correct = 0
                num_total = 0
                fuzzy_total = 0
                field_names = doc.values()

                # split parse by new line for fuzzy evaluation
                split_parse = parse.split('\n')
                for field_name in field_names:
                    fuzzy_total += fuzzy_evaluate(field_name, split_parse)
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
                fuzzy_score = fuzzy_total / len(field_names)
                time_spent = end_time - start_time
                floatAccuracy = (floatCorrect / (floatCorrect+floatWrong)) * 100
                stringAccuracy = (stringCorrect / (stringCorrect + stringWrong)) * 100
                print(doc_name)
                print("Accuracy", accuracy)
                print("float Accuracy", floatAccuracy)
                print("String Accuracy", stringAccuracy)
                print("Fuzzy Score", fuzzy_score)
                print("Time to parse document: {} seconds".format(time_spent))

                accuracy_list.append(accuracy)
                time_list.append(time_spent)
                float_accuracy_list.append(floatAccuracy)
                string_accuracy_list.append(stringAccuracy)
                fuzzy_score_list.append(fuzzy_score)

                writer.writerow([doc_name, accuracy, time_spent,floatAccuracy, stringAccuracy])

                # output images and text to a separate file
                #boxes_output_dir = "/Users/Taaha/Documents/projects"
                boxes_output_dir = "data/boxes/"
                #boxes_output_dir = "/Users/umaymahsultana/Desktop/output"
                if not os.path.exists(boxes_output_dir):
                    os.mkdir(boxes_output_dir)
                create_bounding_boxes(image, file, boxes_output_dir)

                #text_output_dir = "/Users/Taaha/Documents/projects"
                text_output_dir = "data/text/"
                #text_output_dir = "/Users/umaymahsultana/Desktop/output"
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
            fuzzy_score_mean = statistics.mean(fuzzy_score_list)
            writer.writerow(["Average", accuracy_mean, time_mean, float_accuracy_mean, string_accuracy_mean, fuzzy_score_mean])

if __name__ == '__main__':
    evaluate('W2_Clean_DataSet_01_20Sep2019','W2_Truth_and_Noise_DataSet_01.xlsx', 0, 1000, 'Clean', 'W2_Clean_DataSet1_RESULTS.csv')
    evaluate('W2_Noise_DataSet_01_20Sep2019', 'W2_Truth_and_Noise_DataSet_01.xlsx', 1, 1000, 'Noisy','W2_Noisy_DataSet1_RESULTS.csv')
    evaluate('w2_samples_multi_clean', 'W2_Truth_and_Noise_DataSet_02.xlsx', 0, 5000,  'Clean', 'W2_Clean_DataSet2_RESULTS.csv')
    evaluate('w2_samples_multi_noisy', 'W2_Truth_and_Noise_DataSet_02.xlsx', 1, 5000,  'Noisy', 'W2_Noisy_DataSet2_RESULTS.csv')
