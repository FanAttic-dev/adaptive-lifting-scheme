import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2gray

dtype = np.int16

class DWTLevel:
    def __init__(self, img):
        LL, (LH, HL, HH) = lifting_scheme_haar2(img)
        self.coeffs = [LL, LH, HL, HH]
        self.next_level_idx = None
        
    def add_level(self, idx):
        """ Creates a new dwt level at position `idx` and stores it in `self.coeffs`. """
        coeff = self.coeffs[idx]
        
        if len(coeff) <= 1:
            return None
        
        level = DWTLevel(coeff)
        self.coeffs[idx] = level
        self.next_level_idx = idx
        
        return level
    
    def get_next_level(self):
        """ Returns the next dwt level if exists. If the current dwt level is a leaf, returns None """
        next_level_list = [(coeff, idx) for idx, coeff in enumerate(self.coeffs) if isinstance(coeff, DWTLevel)]
        if len(next_level_list) == 0:
            return None, -1
        assert len(next_level_list) == 1
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
    next_level, _ = level.get_next_level()
    if next_level is None:
        return lifting_scheme_haar2_inverse(level.coeffs)
    
    # bottom up
    dwt_level_rec = lifting_scheme_haar2_rec(next_level)
    coeffs = [dwt_level_rec if isinstance(coeff, DWTLevel) else coeff for coeff in level.coeffs]
    return lifting_scheme_haar2_inverse(coeffs)


def visualize_dec(img, root_level):
    def imshow(ax, img):
        ax.imshow(img, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])
        
    
    fig = plt.figure(figsize=(16,8), num=img_name)
    gs = fig.add_gridspec(nrows=1, ncols=2)
    
    ax = fig.add_subplot(gs[0])
    imshow(ax, img)
    
    gs = gs[1].subgridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)
    
    level = root_level
    while True:
        for i, coeff in enumerate(level.coeffs):
            if isinstance(coeff, DWTLevel):
                continue
            
            ax = fig.add_subplot(gs[i])
            imshow(ax, coeff)
        
        next_level, next_level_idx = level.get_next_level()
        if next_level is None:
            break
        else:
            gs = gs[next_level_idx].subgridspec(nrows=2, ncols=2, hspace=0.0, wspace=0.0)
            level = next_level
        
    fig.tight_layout()
    plt.show()


def load_image(path):
    img = skio.imread(path)
    
    # if the image is not grayscale, convert it to grayscale or just take the first channel
    if len(img.shape) == 3 and img.shape[2] > 1:
        try:
            img = rgb2gray(img)
        except:
            img = img[:, :, 0]
        
    return img


def img_noise():
    img = np.zeros((512, 512))
    img = img + np.round(np.random.normal(128, 100, (512, 512))).astype(np.uint8)
    return img
    

images = [
    'additive_noise.png',
    'cameraman.tif', 
    'lake.tif', 
    'lena.tif', 
    'livingroom.tif',
]
img_name = images[0]
img = load_image(f'images/{img_name}')

# Decomposition
root_level = lifting_scheme_haar2_dec(img, max_level=np.inf)

# Reconstruction
img_rec = lifting_scheme_haar2_rec(root_level)

# Visualization of the reconstructed image (left) and the decomposition (right)
visualize_dec(img_rec, root_level)