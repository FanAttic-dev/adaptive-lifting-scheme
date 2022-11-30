import numpy as np
import matplotlib.pyplot as plt
import pywt.data
from numpy import fft as ft
from skimage import io as skio
from math import sqrt

dec_lo, dec_hi, rec_lo, rec_hi = pywt.Wavelet('haar').filter_bank

def my_dwt(f, dec_lo, dec_hi):
    if len(f) % 2 != 0:
        f = np.pad(f, (0, 1), mode='symmetric')
    
    if len(f) > len(dec_lo):
        diff = len(f) - len(dec_lo)
        dec_lo = np.pad(dec_lo, (0, diff))
        dec_hi = np.pad(dec_hi, (0, diff))
    
    #cA = np.convolve(f, dec_lo, mode='same')    
    cA = np.real(ft.ifft(ft.fft(f) * ft.fft(dec_lo)))
    #cD = np.convolve(f, dec_hi, mode='same')
    cD = np.real(ft.ifft(ft.fft(f) * ft.fft(dec_hi)))
    
    startIdx = 1# if len(f) % 2 == 0 else 0
    return cA[startIdx::2], cD[startIdx::2]
    

def my_lifting_scheme_haar(f):
    # split
    A, D = f[::2], f[1::2]
    
    # predict
    D = D - A
    
    # update
    A = A + (D / 2)
    
    # normalize
    A = A * sqrt(2)
    D = D * (sqrt(2) / 2)
    
    return A, D

def my_lifing_scheme_haar_inverse(A, D):
    # nomralize
    A = A * (sqrt(2) / 2)
    D = D * sqrt(2)
    
    A = A - (D / 2)
    D = D + A
    
    res = np.zeros(len(A) + len(D))
    res[::2] = A
    res[1::2] = D
    return res
    

def my_lifting_scheme_haar_dec(f):
    A, D = my_lifting_scheme_haar(f)
    coeffs = [D]
    while len(A) > 1:
        A, D = my_lifting_scheme_haar(A)
        coeffs.append(D)
    
    coeffs.append(A)
    return coeffs
    
        
def my_lifting_scheme_haar_rec(coeffs):
    A = coeffs.pop()
    while coeffs:
        D = coeffs.pop()
        A = my_lifing_scheme_haar_inverse(A, D)
    return A


# single level
f = np.array([1, 4, -3, 0, 1, 2, 3, 4])
A, D = pywt.dwt(f, pywt.Wavelet('haar'))    
pywt.dwt(A, pywt.Wavelet('haar'))    

my_dwt(f, dec_lo, dec_hi)

A, D = my_lifting_scheme_haar(f)
my_lifing_scheme_haar_inverse(A, D)

# full decomposition
coeffs = pywt.wavedec(f, pywt.Wavelet('haar'))
pywt.waverec(coeffs, pywt.Wavelet('haar'))

coeffs = my_lifting_scheme_haar_dec(f)
my_lifting_scheme_haar_rec(coeffs)