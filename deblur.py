import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import color, restoration, io
from scipy.signal import convolve2d as conv2


def motion_kernel(angle, d, sz=65):
    kern = np.ones((1,d), np.float32)
    c = np.cos(angle)
    s = np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2]=(sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5,0))
    kern = cv2.warpAffine(kern, A, (sz,sz), flags=cv2.INTER_CUBIC)
    return kern

# does not work
def find_angle(img):
    f = np.fft.fft2(img)
    fshift= np.fft.fftshift(f)
    magnitude_spectrum = 20*np.long(np.abs(fshift))

    ax= plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True, sharey=True)
    plt.gray()

    ax[0].imshow(magnitude_spectrum)
    ax[0].axis('off')
    ax[0].set_title('spectrum')


def deblur():
    start_time = time.time()

    img = io.imread('/Users/Jisoo/blurred-text.png') #test image
    o_image = color.rgb2gray(img)

    # Trying to find angle
    find_angle(img)

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    image = cv2.filter2D(o_image, -1, sharpen_kernel)

    # PSF (Point Spread Function) motion_kernel( angle , diameter)
    psf = motion_kernel(30, 1)

    #psf = np.ones((3, 3)) / 9

    image_d = conv2(image, psf, 'same')

    deconv = restoration.richardson_lucy(image_d, psf, iterations=40, clip=False)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
    plt.gray()

    print("--- %s seconds ---" % (time.time() - start_time))

    # output window
    ax[0].imshow(o_image)
    ax[0].axis('off')
    ax[0].set_title('original')
    ax[1].imshow(deconv)
    ax[1].axis('off')
    ax[1].set_title('deblur')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    deblur()