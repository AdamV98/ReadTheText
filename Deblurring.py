import cv2 as cv
import numpy as np
import scipy.io as sc

blurred_text = cv.imread("blurred_text.png", cv.IMREAD_GRAYSCALE)
blurred_after_FFT = np.fft.fft2(blurred_text)

data = sc.loadmat("matrices.mat")
Ar = data.get("Ar")
Ac = data.get("Ac")
B = data.get("B")
J = data.get("J")


h, w = B.shape

B_periodic = np.zeros((h*3, w*3), dtype=np.double)
for i in range(3):
    for j in range(3):
        B_periodic[0+i*h:h+i*h, 0+j*w:w+j*w] = B


B_periodic = cv.copyMakeBorder(B, 100, 100, 100, 100, borderType=cv.BORDER_WRAP)

cv.imshow("B_periodic", B_periodic)

matrix = np.zeros((h, w), dtype=np.double)
matrix[int(matrix.shape[0]/2), int(matrix.shape[1]/2)] = 1
psf = np.matmul(Ac, matrix)
psf = np.matmul(psf, np.transpose(Ar))

tmp = np.zeros((h*3, w*3), dtype=np.double)
tmp[h:2*h, w:2*w] = psf
tmp2 = cv.copyMakeBorder(psf, 100, 100, 100, 100, borderType=cv.BORDER_CONSTANT, value=0)
#psf = tmp
psf = tmp2


psf_normalized = cv.normalize(psf, None, 0, 255, cv.NORM_MINMAX)
psf_normalized = np.uint8(psf_normalized)
cv.imshow("psf", psf_normalized)

#P = cv.normalize(P, None, 0, 255, cv.NORM_MINMAX)
#psf = cv.normalize(psf, None, 0, 255, cv.NORM_MINMAX)
#Ar = cv.normalize(Ar, None, 0, 255, cv.NORM_MINMAX)
#Ac = cv.normalize(Ac, None, 0, 255, cv.NORM_MINMAX)

def lucy(img, kernel, iterations):
    blurred = img
    for i in range(iterations):
        img = np.multiply(img, cv.filter2D(np.conj(kernel), -1, np.divide(blurred, cv.filter2D(img, -1, kernel))))

    return img


def wiener_filter(img, kernel, SNR):
    img = np.fft.fft2(img)
    kernel = np.fft.fft2(kernel)
    kernel = np.divide(np.conjugate(kernel),(kernel ** 2 + (1/SNR)))
    img = img * kernel
    img = np.abs(np.fft.ifft2(img))
    return img


deblurred_wiener = wiener_filter(B_periodic, psf, 20000)
deblurred_lucy = lucy(B_periodic, psf, 10)
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
deblurred_lucy = cv.normalize(deblurred_lucy, None, 0, 255, cv.NORM_MINMAX)
deblurred_lucy = np.uint8(deblurred_lucy)
image = cv.medianBlur(image, 5)
cv.imshow("deblurred_wiener", image)
cv.imshow("deblurred_lucy", deblurred_lucy)

#cv.imshow("Ar", Ar)
cv.imshow("B", B)
cv.imwrite("deblurred.png", image)
#cv.imshow("J", J)
#cv.imshow("P", P)
#cv.imshow("PSF", psf)

cv.waitKey(0)
cv.destroyAllWindows()
