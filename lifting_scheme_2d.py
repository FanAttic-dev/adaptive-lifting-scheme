import numpy as np
import matplotlib.pyplot as plt
import pywt.data
from numpy import fft
from skimage import io as skio
import math

def lifting_scheme_haar(f):
    # split
    A, D = f[::2], f[1::2]
    
    # predict
    D -= np.floor(A).astype(dtype)
    
    # update
    A += np.floor(D / 2).astype(dtype)
    
    return A, D

def lifing_scheme_haar_inverse(A, D):
    A -= np.floor(D / 2).astype(dtype)
    D += np.floor(A).astype(dtype)
    
    res = np.zeros(len(A) + len(D))
    res[::2] = A
    res[1::2] = D
    return res
    
def lifting_scheme_haar_dec(f):
    A, D = lifting_scheme_haar(f)
    D_coeffs = [D]
    while len(A) > 1:
        A, D = lifting_scheme_haar(A)
        D_coeffs.append(D)
    
    return A, D_coeffs
    
        
def lifting_scheme_haar_rec(A, D_coeffs):
    while D_coeffs:
        D = D_coeffs.pop()
        A = lifing_scheme_haar_inverse(A, D)
    return A

dtype = np.int16

img = skio.imread('cameraman.tif')
f = img.flatten().astype(dtype)

A, D_coeffs = lifting_scheme_haar_dec(f)
f_rec = lifting_scheme_haar_rec(A, D_coeffs)

img_rec = np.reshape(f_rec, img.shape)
plt.imshow(img_rec, cmap='gray')






fig = plt.figure(figsize=(8,8))

gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)

for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(gs[i])
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()