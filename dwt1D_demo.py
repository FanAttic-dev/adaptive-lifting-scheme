import numpy as np
import matplotlib.pyplot as plt
import pywt.data
from numpy import fft
from skimage import io as skio

dec_lo, dec_hi, rec_lo, rec_hi = pywt.Wavelet('haar').filter_bank
def my_dwt(f, dec_lo, dec_hi):
    # TODO: check f length
    
    # TODO: replace by FFT
    cA = np.convolve(f, dec_lo, mode='same')
    cD = np.convolve(f, dec_hi, mode='same')
    
    return cA[1::2], cD[1::2]

# single level
pywt.dwt([1, 4, -3, 0], pywt.Wavelet('haar'))    
my_dwt([1, 4, -3, 0], dec_lo, dec_hi)

# full decomposition
# TODO: make it cyclic
pywt.wavedec([1, 4, -3, 0], pywt.Wavelet('haar'))
cA, cD = my_dwt([1, 4, -3, 0], dec_lo, dec_hi)
my_dwt(cA, dec_lo, dec_hi), cD
