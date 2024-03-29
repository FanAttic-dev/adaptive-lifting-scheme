import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2gray

def lifting_scheme_haar(f):
    # split
    L, H = f[::2], f[1::2]
    
    # predict
    H -= np.floor(L).astype(dtype)
    
    # update
    L += np.floor(H / 2).astype(dtype)
    
    return L, H

def lifting_scheme_haar2(img):
    f = img.flatten(order='C').astype(dtype)
    L, H = lifting_scheme_haar(f)
    
    new_shape = img.shape[0], img.shape[1] // 2
    L = np.reshape(L, new_shape, order='C').flatten(order='F').astype(dtype)
    H = np.reshape(H, new_shape, order='C').flatten(order='F').astype(dtype)
    
    LL, LH = lifting_scheme_haar(L)
    HL, HH = lifting_scheme_haar(H)
    
    new_shape = new_shape[0] // 2, new_shape[1]
    LL = np.reshape(LL, new_shape, order='F')
    LH = np.reshape(LH, new_shape, order='F')
    HL = np.reshape(HL, new_shape, order='F')
    HH = np.reshape(HH, new_shape, order='F')
    
    return LL, (LH, HL, HH)

def lifting_scheme_haar_inverse(L, H):
    L -= np.floor(H / 2).astype(dtype)
    H += np.floor(L).astype(dtype)
    
    res = np.zeros(len(L) + len(H), dtype=dtype)
    res[::2] = L
    res[1::2] = H
    return res

def lifting_scheme_haar2_inverse(coeffs):
    LL, (LH, HL, HH) = coeffs
    
    new_shape = LL.shape[0] * 2, LL.shape[1]
    
    LL = LL.flatten(order='F').astype(dtype)
    LH = LH.flatten(order='F').astype(dtype)
    HL = HL.flatten(order='F').astype(dtype)
    HH = HH.flatten(order='F').astype(dtype)
    
    L = lifting_scheme_haar_inverse(LL, LH)
    H = lifting_scheme_haar_inverse(HL, HH)
    
    #return np.reshape(L, new_shape, order='F')
    
    L = np.reshape(L, new_shape, order='F').flatten(order='C').astype(dtype)
    H = np.reshape(H, new_shape, order='F').flatten(order='C').astype(dtype)
    
    f_rec = lifting_scheme_haar_inverse(L, H)
    
    new_shape = new_shape[0], new_shape[1] * 2
    img_rec = np.reshape(f_rec, new_shape, order='C')
    
    return img_rec
    
    
def lifting_scheme_haar_dec(img):
    f = img.flatten().astype(dtype)
    
    L, H = lifting_scheme_haar(f)
    
    H_coeffs = [H]
    while len(L) > 1:
        L, H = lifting_scheme_haar(L)
        H_coeffs.append(H)
    
    return L, H_coeffs

def lifting_scheme_haar_dec2(img, max_level=np.inf):
    LL, (LH, HL, HH) = lifting_scheme_haar2(img)
    
    H_coeffs = [(LH, HL, HH)]
    level = 0
    while len(LL) > 1 and level < max_level:
        LL, (LH, HL, HH) = lifting_scheme_haar2(LL)
        H_coeffs.append((LH, HL, HH))
        level += 1
        
    return LL, H_coeffs
    
        
def lifting_scheme_haar_rec(L, H_coeffs):
    while H_coeffs:
        H = H_coeffs.pop()
        L = lifting_scheme_haar_inverse(L, H)
    return L

def visualize_dec(img, max_level=np.inf):
    LL, H_coeffs = lifting_scheme_haar_dec2(img, max_level)
    
    fig = plt.figure(figsize=(8,8))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)
    
    for level, H in enumerate(H_coeffs):
        for i in range(len(H)):
            ax = fig.add_subplot(gs[i+1])
            ax.imshow(H[i], interpolation="nearest", cmap=plt.cm.gray)
            ax.set_xticks([])
            ax.set_yticks([])
            
        if level < max_level:
            gs = gs[0].subgridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)

           
    ax = fig.add_subplot(gs[0])
    ax.imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.tight_layout()
    plt.show()


dtype = np.int16


img = skio.imread('../images/lena_gray_512.tif')
visualize_dec(img)