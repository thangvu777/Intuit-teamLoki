import cv2
import time
import math
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pytesseract
import random
import csv
from pandas import ExcelFile
from fuzzywuzzy import process, fuzz

import locality_aware_nms as nms_locality
import lanms

#tf.app.flags.DEFINE_string('test_data_path', 'data/fake-w2-us-tax-form-dataset/W2_Clean_DataSet_01_20Sep2019', '')
#tf.app.flags.DEFINE_string('test_data_path', 'data/fake-w2-us-tax-form-dataset/realistic/W2_Noise_DataSet_01_20Sep2019', '')
tf.app.flags.DEFINE_string('test_data_path', 'data/fake-w2-us-tax-form-dataset/realistic/w2_samples_multi_noisy', '')

#tf.app.flags.DEFINE_string('test_data_path', 'data/fake-w2-us-tax-form-dataset/w2_samples_multi_clean', '')
#tf.app.flags.DEFINE_string('test_data_path', 'data/fake-w2-us-tax-form-dataset/realistic/W2_Noise_DataSet_01_20Sep2019', '')
#tf.app.flags.DEFINE_string('test_data_path', 'data/fake-w2-us-tax-form-dataset/realistic/w2_samples_multi_noisy', '')

#tf.app.flags.DEFINE_string('results_path', 'tmp/W2_Clean_DataSet1_RESULTS.csv','')
#tf.app.flags.DEFINE_string('results_path', 'tmp/W2_Realistic_Noisy_DataSet1_RESULTS.csv','')
tf.app.flags.DEFINE_string('results_path', 'tmp/W2_Realistic_Noisy_DataSet2_RESULTS.csv','')

tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/Users/thang/utd/cs4485/EAST/tmp/east_icdar2015_resnet_v1_50_rbox', '')
tf.app.flags.DEFINE_string('output_dir', '/Users/thang/utd/cs4485/EAST/tmp/out', '')
tf.app.flags.DEFINE_string('truth_path', 'W2_Truth_Realistic_DataSet_02.xlsx', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def fuzzy_evaluate(field_name:str, parse:str) -> None:
    # TODO: some field names are nan
    # set field names to a blank string if nan
    # if field_name == "nan":
    #     field_name = ""
    extract = process.extractOne(str(field_name), parse)
    # return the score
    return extract[1]


def create_mapping(w2_dir_list: list, truth_file_name_list: list) -> dict:
    # maps w2 folder index to truth excel index
    mapping = set()
    for truth_file_index, truth_file_name in enumerate(truth_file_name_list):
            # noisy data set ends with .jpg or .JPG
            if truth_file_name + '.jpg' in w2_dir_list:
                mapping.add(truth_file_name + '.jpg')

    return mapping

def random_sample(truth_file_name_list:list, truth_docs:list, sample_size:int):
    # need to call the seed before using random to generate the same randon numbers
    random.seed(7)
    sample_file_names = random.sample(truth_file_name_list, sample_size)
    random.seed(7)
    sample_truth_docs = random.sample(truth_docs, sample_size)
    return sample_file_names, sample_truth_docs

def get_images(truth_file_name_list:list, truth_docs: list, sample_size:int):

    inFiles = sorted(os.listdir(FLAGS.test_data_path))
    mapping = create_mapping(inFiles, truth_file_name_list)

    filtered_file_name_list = []
    filtered_truth_docs = []

    for index, file_name in enumerate(truth_file_name_list):
        if file_name + '.jpg' in mapping:
            filtered_file_name_list.append(file_name)
            filtered_truth_docs.append(truth_docs[index])

    random_file_names, sample_truths = random_sample(filtered_file_name_list, filtered_truth_docs, 120)
    print('Size of random sample: ' + str(len(random_file_names)))
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            filename_without_ext = filename.split('.')[0]
            if filename.endswith('.jpg') and filename_without_ext in random_file_names:
                files.append(os.path.join(parent, filename))
    print('Found {} matching images'.format(len(files)))
    return files

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

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

def typeFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
#get boxes
def getBox(box,im):
    h = max(box[2,1],box[3,1]) - min(box[0,1],box[1,1])
    w = max(box[1,0],box[2,0]) - min(box[0,0],box[3,0])
    x = min(box[0,0],box[3,0])
    y = min(box[0,1],box[1,1])
    #print(im.shape)
    pHorizontal = int(w * .2)
    pVertical = int(h * .2)
    pRight = min(x+w+pHorizontal,im.shape[1]-1) #padding horizontal right
    pLeft = max(x-pHorizontal,1)
    pTop = max(y-pVertical,1)
    pBottom = min(y+h+pVertical,im.shape[0]-1)
    #print(pLeft,pRight,pTop,pBottom)
    crop_img = im[pTop:pBottom, pLeft:pRight]
    #print(crop_img.shape)
    #crop_img = im[y-p:y + h + p, x-p:x + w + p]
    #crop_img.size = crop_img.shape
    return crop_img #Image.fromarray(crop_img )


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            dir = '/Users/thang/utd/cs4485/EAST'
            excel_path = os.path.join(dir, FLAGS.truth_path)
            truth_file_name_list, truth_docs = get_excel_docs(excel_path, 0)

            im_fn_list = get_images(truth_file_name_list, truth_docs, 100)

            results_csv = FLAGS.results_path
            with open(results_csv, 'w') as csv_file:
                writer = csv.writer(csv_file)
                # write the headers
                writer.writerow(['Document Name', 'Accuracy', 'Time (seconds)', 'Float Accuracy', 'String Accuracy', 'Fuzzy Score'])
                # Open text file with headers
                with open ("headers.txt", "r") as myFile:
                    headers = myFile.read().split()
                for index, im_fn in enumerate(im_fn_list):
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
                    num_correct = 0
                    num_total = 0

                    fuzzy_score_list = []
                    fuzzy_total = 0

                    allWords = []

                    im = cv2.imread(im_fn)[:, :, ::-1]
                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = resize_image(im)

                    timer = {'net': 0, 'restore': 0, 'nms': 0}
                    start = time.time()
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                    timer['net'] = time.time() - start

                    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                    print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                        im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h

                    duration = time.time() - start_time
                    print('[timing] {}'.format(duration))

                    #truth
                    truth_doc = truth_docs[index]
                    field_names = truth_doc.values()
                    start_time = time.time()

                    # save to file
                    if boxes is not None:
                        res_file = os.path.join(
                            FLAGS.output_dir,
                            '{}.txt'.format(
                                os.path.basename(im_fn).split('.')[0]))
                        with open(res_file, 'w') as f:
                            for box in boxes:
                                # to avoid submitting errors
                                box = sort_poly(box.astype(np.int32))
                                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                    continue
                                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                                ))

                                crop_img = getBox(box,im)
                                #--oem 1
                                output = pytesseract.image_to_string(crop_img, config='--psm 6')

                                #output = pytesseract.image_to_string(crop_img)
                                allWords.append(output)
                                #cv2.imshow("box",crop_img)
                                #cv2.waitKey(0)

                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

                    end_time = time.time()
                    try:
                        allWords.remove('')
                    except:
                        print("No empty spaces")

                    parse = ' '.join(allWords)
                    if (parse != ''):
                        for field_name in field_names:
                            fuzzy_total += fuzzy_evaluate(field_name, parse)
                            if str(field_name) in parse:
                                num_correct += 1
                                if (typeFloat(str(field_name))):
                                    floatCorrect += 1
                                else:
                                    stringCorrect += 1
                                    parse.replace(str(field_name), '', 1)
                            else:
                                if (typeFloat(str(field_name))):
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
                    else:
                        accuracy = 0
                        fuzzy_score = 0
                        time_spent = end_time - start_time
                        floatAccuracy = 0
                        stringAccuracy = 0

                    print("Accuracy", accuracy)
                    print("float Accuracy", floatAccuracy)
                    print("String Accuracy", stringAccuracy)
                    print("Fuzzy Score", fuzzy_score)
                    print("Time to parse document: {} seconds".format(time_spent))
                    print("==================================================================")
                    accuracy_list.append(accuracy)
                    time_list.append(time_spent)
                    float_accuracy_list.append(floatAccuracy)
                    string_accuracy_list.append(stringAccuracy)
                    fuzzy_score_list.append(fuzzy_score)
                    writer.writerow([im_fn, accuracy, time_spent, floatAccuracy, stringAccuracy, fuzzy_score])
                    if not FLAGS.no_write_images:
                        img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                        cv2.imwrite(img_path, im[:, :, ::-1])

if __name__ == '__main__':
    tf.app.run()

