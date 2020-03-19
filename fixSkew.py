import pytesseract
from pytesseract import Output
import time
import cv2
import os
import numpy as np

def fix_skew(img: str):
    skew_output = 'data/skew/fix_skew_output/'
    img = cv2.imread(img)
    # convert the image to grayscale and flip the foreground
    # background -> black; foreground -> white
    # background pixels = 0; foreground pixels = 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
	    angle = -angle
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Input", img)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)

def process ( directory : str , fix_skew : bool) -> None:
    for filename in sorted(os.listdir(directory)):
        start_time = time.time()
        if filename.endswith('.jpg'):
            file = os.path.join(directory, filename)
            print('OUTPUT of %s: ' %(file))
            print(pytesseract.image_to_string(file))
            save = filename[:-4] + '_boxes' + '.jpg'
            #create_bounding_boxes(save, file, 'truncated/out')
        else:
            continue
        end_time = time.time()
        total = end_time - start_time
        print('========================================================')
        print('%s took %.2f seconds to finish.' % (filename, total))
        print('========================================================')

def detect_outline(img:str)-> None:
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

    # find contours
    cntrs = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # find biggest contour
    area_thresh = 0
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c

    # draw the contour on a copy of the input image
    results = img.copy()
    cv2.drawContours(results, [big_contour], 0, (0, 0, 255), 2)

    # write result to disk
    #cv2.imwrite(img + '_thresh', thresh)
    #cv2.imwrite(img + '_results', results)

    #cv2.imshow("THRESH", thresh)
    cv2.imshow("RESULTS", results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    dir = 'data/skewed/W2_XL_input_noisy_1017.jpg'
    #dir = 'data/skewed/W2_Multi_Sample_Data_input_ADP1_clean_15601.jpg'
    #dir = 'data/skewed/8.png'
    detect_outline(dir)