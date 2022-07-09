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
P = data.get("P")
psf = data.get("PSF")

#P = cv.normalize(P, None, 0, 255, cv.NORM_MINMAX)
#psf = cv.normalize(psf, None, 0, 255, cv.NORM_MINMAX)
#Ar = cv.normalize(Ar, None, 0, 255, cv.NORM_MINMAX)
#Ac = cv.normalize(Ac, None, 0, 255, cv.NORM_MINMAX)

def wiener_filter(img, kernel, SNR):
    img = np.fft.fft2(img)
    kernel = np.fft.fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + (1/SNR))
    img = img * kernel
    img = np.abs(np.fft.ifft2(img))
    return img


deblurred = wiener_filter(B, P, 10000)
h, w = deblurred.shape
deblurred1 = deblurred[int(h/2):h, int(w/2):w]
deblurred2 = deblurred[int(h/2):h, 0:int(w/2)]
deblurred3 = deblurred[0:int(h/2), int(w/2):w]
deblurred4 = deblurred[0:int(h/2), 0:int(w/2)]
image = np.zeros(deblurred.shape)
image[0:int(h/2), 0:int(w/2)] = deblurred1
image[0:int(h/2), int(w/2):w] = deblurred2
image[int(h/2):h, 0:int(w/2)] = deblurred3
image[int(h/2):h, int(w/2):w] = deblurred4
image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
image = np.uint8(image)
cv.imshow("deblurred", image)

#cv.imshow("Ar", Ar)
cv.imshow("B", B)
#cv.imshow("J", J)
#cv.imshow("P", P)
#cv.imshow("PSF", psf)

cv.waitKey(0)
cv.destroyAllWindows()
