import cv2 as cv
import numpy as np
import scipy.io as sc

data = sc.loadmat("matrices.mat")
Ar = data.get("Ar")
Ac = data.get("Ac")
B = data.get("B")
J = data.get("J")

h, w = B.shape

matrix = np.zeros((h, w), dtype=np.double)
matrix[int(matrix.shape[0]/2), int(matrix.shape[1]/2)] = 1
psf = np.matmul(Ac, matrix)
psf = np.matmul(psf, np.transpose(Ar))
#psf = tmp

psf_normalized = cv.normalize(psf, None, 0, 255, cv.NORM_MINMAX)
psf_normalized = np.uint8(psf_normalized)
cv.imshow("psf", psf_normalized)

def lucy(img, kernel, iterations):
    blurred = img
    for i in range(iterations):
        img = np.multiply(img, cv.filter2D(np.conj(kernel), -1, np.divide(blurred, cv.filter2D(img, -1, kernel))))

    return img

def inverse_filter(B, psf):
    B_fft = np.fft.fft2(B)
    psf_fft = np.fft.fft2(psf)

    image_fft = B_fft / psf_fft

    image = np.abs(np.fft.ifft2(image_fft))

    return image

def tikhonov(B, Ar, Ac, alpha):
    Uc, Sc, Vc = np.linalg.svd(Ac)
    Ur, Sr, Vr = np.linalg.svd(Ar)

    S = np.atleast_2d(Sc).T @ np.atleast_2d(Sr)

    phi = np.divide(np.power(np.abs(S), 2), np.power(np.abs(S), 2)+alpha**2)
    Sfilt = phi / S
    tmp = np.matmul(Uc.T, B)
    tmp = np.matmul(tmp, Ur)
    tmp2 = tmp * Sfilt
    Xfilt = np.matmul(Vc.T, tmp2)
    Xfilt = np.matmul(Xfilt, Vr)

    return Xfilt


def wiener_filter(img, kernel, snr):
    """
    Applies the Wiener filter according to the given parameters

    Parameters
    ----------
        img : ndarray
            image in numpy array format
        kernel : ndarray
            the kernel for the wiener filter
        snr : int
            signal to noise ratio

    Returns
    -------
        img
            the return image
    """
    img = np.fft.fft2(img)
    kernel = np.fft.fft2(kernel)
    kernel = np.divide(np.conjugate(kernel), (kernel ** 2 + (1/snr)))
    img = img * kernel
    img = np.abs(np.fft.ifft2(img))
    return img


deblurred_wiener = wiener_filter(B, psf, 20000)
deblurred_inverse = inverse_filter(B, psf)
deblurred_tikhonov = tikhonov(B, Ar, Ac, 0.002)
h, w = deblurred_wiener.shape
deblurred1 = deblurred_wiener[int(h / 2):h, int(w / 2):w]
deblurred2 = deblurred_wiener[int(h / 2):h, 0:int(w / 2)]
deblurred3 = deblurred_wiener[0:int(h / 2), int(w / 2):w]
deblurred4 = deblurred_wiener[0:int(h / 2), 0:int(w / 2)]
image = np.zeros(deblurred_wiener.shape)
image[0:int(h/2), 0:int(w/2)] = deblurred1
image[0:int(h/2), int(w/2):w] = deblurred2
image[int(h/2):h, 0:int(w/2)] = deblurred3
image[int(h/2):h, int(w/2):w] = deblurred4
image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
image = np.uint8(image)
deblurred_inverse = cv.normalize(deblurred_inverse, None, 0, 255, cv.NORM_MINMAX)
deblurred_inverse = np.uint8(deblurred_inverse)
image = cv.medianBlur(image, 5)
cv.imshow("deblurred_wiener", image)
cv.imshow("deblurred_inverse", deblurred_inverse)
cv.imshow("deblurred_tikhonov", deblurred_tikhonov)

cv.imshow("B", B)

cv.waitKey(0)
cv.destroyAllWindows()
