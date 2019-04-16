import numpy as np
from PIL import Image
from io import BytesIO

def jpeg_compression(img, quality=75):
    with BytesIO() as f:
        img_obj = Image.fromarray(img)
        img_obj.save(f, format='JPEG', quality=quality)
        f.seek(0)
        img_obj = Image.open(f)
        return np.asarray(img_obj)
    
class JPEG(object):
    def __init__(self, q=75):
        self.q = q

    def __call__(self, img):
        img = np.asarray(img)
        if (img.dtype == np.float32 or img.dtype == np.float64) and img.max() <= 1 and img.min() >= 0:
            img = (255*img).astype(np.uint8)
        res = jpeg_compression(img, quality=self.q)
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(quality={0})'.format(self.q)   