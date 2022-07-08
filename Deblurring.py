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

P = cv.normalize(P, None, 0, 255, cv.NORM_MINMAX)

cv.imshow("Ar", Ar)
cv.imshow("B", B)
cv.imshow("P", P)

cv.waitKey(0)

cv.destroyAllWindows()
