import cv2 as cv
import numpy as np
import scipy.io as sc


def normalize_uint8(img):
    """
    Normalizes a floating point image to 8 bit unsigned integer format

    Parameters
    ----------
        img : ndarray
            image to be normalized

    Returns
    -------
        norm_img
            the return image
    """
    norm_img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    norm_img = np.uint8(norm_img)

    return norm_img


def inverse_filter(B, psf):
    """
    Applies the inverse filter according to the given parameters

    Parameters
    ----------
        B : ndarray
            blurred image in numpy array format
        psf : ndarray
            the point spread function of the applied convolution

    Returns
    -------
        image
            the return image
    """
    B_fft = np.fft.fft2(B)
    psf_fft = np.fft.fft2(psf)

    image_fft = B_fft / psf_fft

    image = np.abs(np.fft.ifft2(image_fft))

    return image


def tikhonov(B, Ar, Ac, alpha):
    """
    Applies the Tikhonov method according to the given parameters to the blurred image

    Parameters
    ----------
        B : ndarray
            the blurred image in numpy array format
        Ar : ndarray
            row component of the applied convolution
        Ac : ndarray
            column component of the applied convolution
        alpha : float
            the alpha value for the Tikhonov method

    Returns
    -------
        img
            the return image
    """

    Uc, Sc, Vc = np.linalg.svd(Ac)
    Ur, Sr, Vr = np.linalg.svd(Ar)

    S = np.atleast_2d(Sc).T @ np.atleast_2d(Sr)
    phi = np.divide(np.power(np.abs(S), 2), np.power(np.abs(S), 2)+alpha**2)
    Sfilt = phi / S
    Xfilt = Vc.T @ ((Uc.T @ B @ Ur) * Sfilt) @ Vr

    return Xfilt


def wiener_filter(B, psf, snr):
    """
    Applies the Wiener filter according to the given parameters

    Parameters
    ----------
        B : ndarray
            the blurred image in numpy array format
        psf : ndarray
            the point spread function of the applied convolution
        snr : int
            signal to noise ratio

    Returns
    -------
        deblurred
            the return image
    """

    img = np.fft.fft2(B)
    kernel = np.fft.fft2(psf)
    kernel = np.divide(np.conjugate(kernel), (np.abs(kernel) ** 2 + (1/snr)))
    img = img * kernel
    img = np.abs(np.fft.ifft2(img))

    deblurred = rearrange_image(img)

    return deblurred


def rearrange_image(img):
    """
    Rearranges the image to be in the correct order

    Parameters
    ----------
        img : ndarray
            the image that is going to be rearranged

    Returns
    -------
        rearranged_img
            the rearranged image
    """
    h, w = img.shape
    img1 = img[int(h / 2):h, int(w / 2):w]
    img2 = img[int(h / 2):h, 0:int(w / 2)]
    img3 = img[0:int(h / 2), int(w / 2):w]
    img4 = img[0:int(h / 2), 0:int(w / 2)]
    rearranged_img = np.zeros(img.shape)
    rearranged_img[0:int(h / 2), 0:int(w / 2)] = img1
    rearranged_img[0:int(h / 2), int(w / 2):w] = img2
    rearranged_img[int(h / 2):h, 0:int(w / 2)] = img3
    rearranged_img[int(h / 2):h, int(w / 2):w] = img4

    return rearranged_img


def create_psf(B, Ar, Ac):
    """
    Postprocess function

    Parameters
    ----------
        B : ndarray
            the blurred image
        Ar : ndarray
            the row component of the psf
        Ac : ndarray
            the column component of the psf

    Returns
    -------
        psf
            the point spread function
    """

    h, w = B.shape
    dirac = np.zeros((h, w), dtype=np.double)
    dirac[int(dirac.shape[0] / 2), int(dirac.shape[1] / 2)] = 1
    psf = Ac @ dirac @ Ar.T

    return psf


def postprocess(image):
    """
    Postprocess function

    Parameters
    ----------
        image : ndarray
            the image that the post processing will be applied to

    Returns
    -------
        img
            the return image
    """

    filter = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image = cv.filter2D(image, -1, filter)

    _, image = cv.threshold(image, 180, 255, cv.THRESH_TRUNC)
    clahe = cv.createCLAHE(clipLimit=30)
    image = clahe.apply(image) + 1

    return image

# reading the matrices from the Matlab file
data = sc.loadmat("matrices.mat")
Ar = data.get("Ar")
Ac = data.get("Ac")
B = data.get("B")

psf = create_psf(B, Ar, Ac)

cv.imshow("point spread function", normalize_uint8(psf))
cv.imshow("blurred image", B)

deblurred_wiener = wiener_filter(B, psf, 20000)
deblurred_inverse = inverse_filter(B, psf)
deblurred_tikhonov = tikhonov(B, Ar, Ac, 0.002)

deblurred_tikhonov = normalize_uint8(deblurred_tikhonov)

deblurred_tikhonov = postprocess(deblurred_tikhonov)

cv.imshow("deblurred_wiener", normalize_uint8(deblurred_wiener))
cv.imshow("deblurred_inverse", normalize_uint8(deblurred_inverse))
cv.imshow("deblurred_tikhonov", deblurred_tikhonov)
#cv.imshow("deblurred_tikhonov_thinned", deblurred_tikhonov_thinned)

cv.imwrite("blurred.png", normalize_uint8(B))
cv.imwrite("deblurred_wiener.png", normalize_uint8(deblurred_wiener))
cv.imwrite("deblurred_inverse.png", normalize_uint8(deblurred_inverse))
cv.imwrite("deblurred_tikhonov.png", deblurred_tikhonov)
cv.imwrite("psf.png", normalize_uint8(psf))

cv.waitKey(0)
cv.destroyAllWindows()
