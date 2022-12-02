import numpy as np
import matplotlib.pyplot as plt
import pywt.data
from numpy import fft
from skimage import io as skio

img = skio.imread('../images/cameraman.tif')
#img = pywt.data.camera()

coeffs = pywt.dwt2(img, 'coif1')
LL, (LH, HL, HH) = coeffs

fig = plt.figure(figsize=(8,8))

gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)

gs01 = gs[0, 0].subgridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(gs01[i])
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])
    

for i, a in enumerate([LL, LH, HL, HH]):
    if i == 0:
        continue
    ax = fig.add_subplot(gs[i])
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()