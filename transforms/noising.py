import numpy as np
import torch
import torch.nn.functional as F
import math, random
from PIL import Image


# POISSON-GAUSSIAN DENOISING USING THE EXACT UNBIASED INVERSE OF THEGENERALIZED ANSCOMBE TRANSFORMATION
def add_poisson_gaussian_noise(x, a, b):
    if a==0:
        z = x + math.sqrt(b)*torch.randn(*x.shape)
    else:
        chi = 1/a
        z = torch.poisson(chi*x)/chi + math.sqrt(b)*torch.randn(*x.shape)
    return torch.max(torch.zeros_like(z), torch.min(torch.ones_like(z), z))


class PoissonGaussianNoise(object):
    def __init__(self, a, b, threshold = 0.5):
        self.a = a
        self.b = b
        self.threshold = threshold

    def __call__(self, img):
        if random.random() > self.threshold:
            return img
        return add_poisson_gaussian_noise(img, self.a, self.b)

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1})'.format(self.a, self.b)


class GaussianNoise(object):
    def __init__(self, std, threshold = 0.5, clamp=None, mean=0):
        self.std = std
        self.mean = mean
        self.threshold = threshold
        self.clamp = clamp

    def __call__(self, img):
        if random.random() > self.threshold: # noising with the probability as threshold
            return img
        
        if isinstance(self.std, list):
            std = np.random.choice(self.std)
        else:
            std = self.std

        img = img + torch.randn(*img.shape)*std + self.mean
        
        if self.clamp is None: #clamp is specifiying lower and upper bounds
            return img
        else:
            return torch.clamp(img, *self.clamp)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

def bilinear(y):
    r""" Initialize with bilinear interpolation"""
    F_r = torch.FloatTensor([[1,2,1],[2,4,2],[1,2,1]])/4
    F_b = F_r
    F_g = torch.FloatTensor([[0,1,0],[1,4,1],[0,1,0]])/4
    bilinear_filter = torch.stack([F_r,F_g,F_b])[:,None]
    if y.is_cuda:
        bilinear_filter = bilinear_filter.cuda()
    res = F.conv2d(y, bilinear_filter,padding=1, groups=3)
    return res

# def generate_mask(im_shape, pattern='RGGB'):
#     if pattern == 'RGGB':
#         # pattern RGGB
#         r_mask = torch.zeros(im_shape)
#         r_mask[0::2, 0::2] = 1

#         g_mask = torch.zeros(im_shape)
#         g_mask[::2, 1::2] = 1
#         g_mask[1::2, ::2] = 1

#         b_mask = torch.zeros(im_shape)
#         b_mask[1::2, 1::2] = 1

#         mask = torch.zeros(im_shape + (3,))
#         mask[:, :, 0] = r_mask
#         mask[:, :, 1] = g_mask
#         mask[:, :, 2] = b_mask
#         return mask

def generate_mask(im_shape, pattern='RGGB'):
    if pattern == 'RGGB':
        # pattern RGGB
        r_mask = np.zeros(im_shape)
        r_mask[0::2, 0::2] = 1
        
        g_mask = np.zeros(im_shape)
        g_mask[::2, 1::2] = 1
        g_mask[1::2, ::2] = 1
        
        b_mask = np.zeros(im_shape)
        b_mask[1::2, 1::2] = 1

        mask = np.zeros(im_shape + (3,))
        mask[:, :, 0] = r_mask
        mask[:, :, 1] = g_mask
        mask[:, :, 2] = b_mask
        return mask
    
# class RawPoissonGaussianNoise(object):
#     def __init__(self, a, b, threshold = 0.5):
#         self.a = a
#         self.b = b
#         self.threshold = threshold

#     def __call__(self, img):
#         if random.random() > self.threshold:
#             return img
#         img = np.asarray(img)
#         h, w, c = img.shape
#         raw_img = generate_mask((h, w))*img
#         noised = add_poisson_gaussian_noise(torch.from_numpy(raw_img).float(), self.a, self.b)
#         noised = noised.permute(2,0,1)
#         demosaicked = bilinear(noised.unsqueeze(0)).squeeze(0).permute(1,2,0).numpy()
#         return demosaicked

def realistic_noise(img, a, b):
    assert (img.min() >= 0 and img.max() <=1), 'image range should be between 0 and 1'
    y = np.random.normal(loc=img, scale=a*img+b**2)
    if img.dtype == np.uint8:
        y = y.astype(np.uint8)
    elif img.dtype == np.float32 or img.dtype == np.float64:
#         y = y.astype(img.dtype).clip(0,1)
        y = y.astype(np.float32).clip(0,1)
    return y

def srgb_to_linrgb(img):
    """ Convert sRGB color space to linRGB 
        https://en.wikipedia.org/wiki/SRGB
    """
    assert img.dtype in [np.float32, np.float64] 
    img = img.copy()
    mask = img <= 0.04045
    img[~mask] = ((img[~mask]+0.055)/1.055)**2.4
    img[mask] = img[mask] / 12.92
    return img

def linrgb_to_srgb(img):
    """ Convert linRGB color space to sRGB 
        https://en.wikipedia.org/wiki/SRGB
    """
    assert img.dtype in [np.float32, np.float64] 
    img = img.copy()
    mask = img <= 0.0031308
    img[~mask] = (img[~mask]**(1/2.4))*(1.055) - 0.055
    img[mask] = img[mask] * 12.92
    return img

class RawNoise(object):
    def __init__(self, a, b, threshold = 0.5, img_scaler = 1):
        self.a = a
        self.b = b
        self.threshold = threshold
        self.img_scaler = img_scaler

    def __call__(self, img):
        if random.random() > self.threshold:
            return img
        
        img = np.asarray(img)
        if img.dtype == np.uint8:
            img = img/255.
        assert img.dtype == np.float32 or img.dtype == np.float64, 'image should be np.float32 or np.float64'
        assert (img.min() >= 0 and img.max() <=1), 'image range should be between 0 and 1'
        
        h, w, c = img.shape
        # img_np = linrgb_to_srgb(img_np)
        img = srgb_to_linrgb(img*self.img_scaler)
        raw_img = generate_mask((h,w))*img

        img_noised = realistic_noise(raw_img, self.a, self.b)
        demosaicked = bilinear(torch.from_numpy(img_noised).float().permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0).numpy()
        # demosaicked_np = srgb_to_linrgb(demosaicked_np).clip(0,1)
        demosaicked = linrgb_to_srgb(demosaicked/self.img_scaler).clip(0,1)
        demosaicked = Image.fromarray((demosaicked*255).astype(np.uint8))
        return demosaicked

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1})'.format(self.a, self.b)


class RawNoiseBayer(object):
    def __init__(self, a, b, threshold = 0.5, scaler = 1):
        self.a = a
        self.b = b
        self.threshold = threshold
        self.scaler = scaler

    def __call__(self, img):
        img = np.asarray(img)
        if img.dtype == np.uint8:
            img = (img/255.).astype(np.float32)
        assert img.dtype == np.float32 or img.dtype == np.float64, 'image should be np.float32 or np.float64'
        assert (img.min() >= 0 and img.max() <=1), 'image range should be between 0 and 1'
        h, w, c = img.shape
        img = img*self.scaler
        img = srgb_to_linrgb(img)
        raw_img = generate_mask((h,w))*img

        if random.random() > self.threshold:
            return raw_img.astype(np.float32)
        
        return realistic_noise(raw_img, self.a, self.b).clip(0,1)

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1})'.format(self.a, self.b)


class RawPoissonGaussianNoise(object):
    def __init__(self, a, b, threshold = 0.5):
        self.a = a
        self.b = b
        self.threshold = threshold

    def __call__(self, img):
        if random.random() > self.threshold:
            return img
        c, h, w = img.shape
        raw_img = generate_mask((h, w))*img
        noised = add_poisson_gaussian_noise(traw_img, self.a, self.b)
        demosaicked = bilinear(noised.unsqueeze(0)).squeeze(0)
        return demosaicked

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1})'.format(self.a, self.b)
    
# class RawPoissonGaussianNoise(object):
#     def __init__(self, a, b, threshold = 0.5):
#         self.a = a
#         self.b = b
#         self.threshold = threshold

#     def __call__(self, img):
#         if random.random() > self.threshold:
#             return img
#         c, h, w = img.shape
#         raw_img = generate_mask((h, w)).permute(2,0,1)*img
#         noised = add_poisson_gaussian_noise(raw_img, self.a, self.b)
#         demosaicked = bilinear(noised.unsqueeze(0)).squeeze(0)
#         return demosaicked

#     def __repr__(self):
#         return self.__class__.__name__ + '(a={0}, b={1})'.format(self.a, self.b)