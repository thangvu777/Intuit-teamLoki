import pytesseract
from pytesseract import Output
import time
import cv2
import os
import numpy as np
import math
import PIL

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

def process (directory : str, outD:str) -> None:
    for filename in sorted(os.listdir(directory)):
        start_time = time.time()
        if filename.endswith('.jpg'):
            file = os.path.join(directory, filename)
            print('Fixing Skew of %s: ' %(file))
            img = fix_skew(file)
            cv2.imwrite(os.path.join(outD, filename))
        else:
            continue
        end_time = time.time()
        total = end_time - start_time
        print('========================================================')
        print('%s took %.2f seconds to finish.' % (filename, total))
        print('========================================================')

# Fix text skew of an image
def fix_skew(img: str):
    fix_skew_helper(img)

''' 
    Fix skew helper:
    1. Find the biggest contour outline of a given image
    2. Get the dimensions of the image using shape
    3. Find the approximate Cartesian coordinates of the image  
    4. Calculate the angle between 3 points (2 points from the image, 1 from (0,0))
    5. Rotate and crop the image using the center of the original image
'''
def fix_skew_helper(img: str):
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

    ''' # Uncomment if you want to see the contour lines
    # (RED) draw the contour on a copy of the input image
    results = img.copy()
    cv2.drawContours(results, [big_contour], 0, (0, 0, 255), 2)
    cv2.imshow('results', results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # STEP 2: get dimensions of image
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]

    # STEP 3: Find approximate coordinates of outlining box
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
        print(x,y)

    # FUNCTION DEFINITION: Calculate an angle given 3 Cartesian coordinates
    def getAngle(a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang

    # STEP 4: Calculate degree of rotation
    angle = (getAngle([0,0], points[0], points[2])) - 360

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

    # STEP 5: Rotate and Crop
    rotated_image = rotate(img, center=(width/2, height/2), theta=angle, width= width, height=height)

    ''' # Uncomment if you want to see the rotated image
    cv2.imshow('rotated',rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # Return the final rotated image
    return rotated_image

if __name__ == '__main__':
    dir = 'data/fake-w2-us-tax-form-dataset/w2_samples_multi_noisy'
    outD = 'data/skewed/rotatefix'
    if not os.path.exists(outD):
        os.mkdir(outD)
    process(dir, outD)

    cv2.imwrite(os.path.join(output_dir, image_name), img)