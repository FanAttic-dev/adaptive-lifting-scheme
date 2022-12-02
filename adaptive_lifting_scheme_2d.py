import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio


class DWTLevel:
    def __init__(self, img):
        LL, (LH, HL, HH) = lifting_scheme_haar2(img)
        self.coeffs = [LL, LH, HL, HH]
        
    def add_level(self, idx):
        """ Creates a new dwt level at position `idx` and stores it in `self.coeffs`. """
        coeff = self.coeffs[idx]
        
        if len(coeff) <= 1:
            return None
        
        level = DWTLevel(coeff)
        self.coeffs[idx] = level
        
        return level
    
    def get_next_level(self):
        """ Returns the next dwt level if exists. If the current dwt level is a leaf, returns None """
        next_level_list = [coeff for coeff in self.coeffs if isinstance(coeff, DWTLevel)]
        if len(next_level_list) == 0:
            return None
        return next_level_list[0]
        
    def get_max_var_coeff_idx(self):
        """ Returns the index of the coefficient with the maximal variance. """
        return np.argmax(list(map(lambda coeff: np.var(coeff), self.coeffs)))
    

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
    # update
    L -= np.floor(H / 2).astype(dtype)
    
    # predict
    H += np.floor(L).astype(dtype)
    
    # merge
    res = np.zeros(len(L) + len(H), dtype=dtype)
    res[::2] = L
    res[1::2] = H
    return res


def lifting_scheme_haar2_inverse(coeffs):
    if isinstance(coeffs, list):
        LL, LH, HL, HH = coeffs
    elif isinstance(coeffs, tuple):
        LL, (LH, HL, HH) = coeffs        
    
    new_shape = LL.shape[0] * 2, LL.shape[1]
    
    LL = LL.flatten(order='F').astype(dtype)
    LH = LH.flatten(order='F').astype(dtype)
    HL = HL.flatten(order='F').astype(dtype)
    HH = HH.flatten(order='F').astype(dtype)
    
    L = lifting_scheme_haar_inverse(LL, LH)
    H = lifting_scheme_haar_inverse(HL, HH)
    
    L = np.reshape(L, new_shape, order='F').flatten(order='C').astype(dtype)
    H = np.reshape(H, new_shape, order='F').flatten(order='C').astype(dtype)
    
    f_rec = lifting_scheme_haar_inverse(L, H)
    
    new_shape = new_shape[0], new_shape[1] * 2
    img_rec = np.reshape(f_rec, new_shape, order='C')
    
    return img_rec
        

def lifting_scheme_haar2_dec(img, max_level=np.inf):
    """ Performs DWT decomposition based on variance using lifting scheme of `img` up to level `max_level`. """
    
    # TODO: perform padding
    
    root = DWTLevel(img)
    
    dwt_level = root
    level = 0
    while dwt_level is not None and level < max_level:
        max_var_coeff_idx = dwt_level.get_max_var_coeff_idx()
        dwt_level = dwt_level.add_level(max_var_coeff_idx)
        level += 1
    
    return root


def lifting_scheme_haar2_rec(level):
    """ Performs DWT reconstruction recursively of a given root level `level`. """
    
    # top down
    next_level = level.get_next_level()
    if next_level is None:
        return lifting_scheme_haar2_inverse(level.coeffs)
    
    # bottom up
    dwt_level_rec = lifting_scheme_haar2_rec(next_level)
    coeffs = [dwt_level_rec if isinstance(coeff, DWTLevel) else coeff for coeff in level.coeffs]
    return lifting_scheme_haar2_inverse(coeffs)


def visualize_dec(root_level):
    fig = plt.figure(figsize=(8,8))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)
    
    level = root_level
    while True:
        next_level = None
        for i, coeff in enumerate(level.coeffs):
            if isinstance(coeff, DWTLevel):
                next_level = coeff
                continue
            
            ax = fig.add_subplot(gs[i])
            ax.imshow(coeff, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_xticks([])
            ax.set_yticks([])
        
        if next_level is None:
            break
        else:
            gs = gs[level.coeffs.index(next_level)].subgridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)
            level = next_level
        
    fig.tight_layout()
    plt.show()


dtype = np.int16
img = skio.imread('images/cameraman.tif')
root_level = lifting_scheme_haar2_dec(img, max_level=3)
img_rec = lifting_scheme_haar2_rec(root_level)
#plt.imshow(img_rec, cmap='gray')
visualize_dec(root_level)