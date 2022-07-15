# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:59:49 2022
"""

import cv2 as cv
import numpy as np
import scipy.io as sc
import matplotlib.pyplot as plt

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


def compare_2_imgs (img1, img2, title1 = 'Image 1', title2 = 'Image2', size = (15,15), title_size = 15, cmap = 'gray'):
     
    # create figure
    fig = plt.figure(figsize=size)
      
    # setting values to rows and column variables
    rows = 1
    columns = 2
      
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
      
    # showing image
    plt.imshow(img1, cmap = cmap)
    plt.axis('off')
    plt.title(title1, fontsize = title_size)
      
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
      
    # showing image
    plt.imshow(img2, cmap = cmap)
    plt.axis('off')
    plt.title(title2, fontsize = title_size)


    
    










