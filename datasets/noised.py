import torch
import random
from torchvision import datasets
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class NoisedDataset(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dirs, transform=None, noise_transform=None):

        super(NoisedDataset, self).__init__(dirs, transform)
        self.noise_transform = noise_transform

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            
            if self.transform is not None:
                return self.transform(img)
            return img

        img_path, class_id = self.imgs[index]
        img1 = transform(img_path)
        img2 = self.noise_transform(img1)
        
        return 255*img2, 255*img1, class_id

to_tensor = transforms.ToTensor()
class NoisedBayerDataset(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dirs, transform=None, noise_transform=None, scaler=1):

        super(NoisedBayerDataset, self).__init__(dirs, transform)
        self.noise_transform = noise_transform
        self.scaler = scaler

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            
            if self.transform is not None:
                return self.transform(img)
            return img

        img_path, class_id = self.imgs[index]
        img1 = transform(img_path)
        img1 = Image.fromarray((self.scaler*np.asarray(img1)).astype(np.uint8))
        img2 = self.noise_transform(img1) #.sum(axis=2, keepdims=True)
        path = os.path.relpath(img_path, self.root)
        return 255*to_tensor(img2), 255*to_tensor(img1), class_id, path


class NoisedDataset_w_dirs(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dirs, transform=None, noise_transform=None):

        super(NoisedDataset_w_dirs, self).__init__(dirs, transform)
        self.noise_transform = noise_transform

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            
            if self.transform is not None:
                return self.transform(img)
            return img

        img_path, class_id = self.imgs[index]
        img1 = transform(img_path)
        img2 = self.noise_transform(img1)
        img1, img2 = 255*img1, 255*img2
        path = os.path.relpath(img_path, self.root)
        return img2, img1, class_id, path
    
    
    
class NoisedDataset_w_storage(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dirs, transform=None, noise_transform=None, storage_dir=None):

        super(NoisedDataset_w_storage, self).__init__(dirs, transform)
        self.noise_transform = noise_transform
        self.storage_dir = storage_dir

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''
        img_path, class_id = self.imgs[index]
        img1 = self.loader(img_path)
        
        use_current_img = True
        if self.storage_dir is not None:
            img_path_storage = os.path.join(self.storage_dir, os.path.relpath(img_path, self.root))
            
            if os.path.exists(img_path_storage):
                img2 = self.loader(img_path_storage)
                use_current_img = False

        if use_current_img:
            img2 = self.noise_transform(np.asarray(img1))
            img2 = Image.fromarray(img2.clip(0, 255).astype(np.uint8))
       
        if self.storage_dir is not None and not os.path.exists(img_path_storage):
            os.makedirs(os.path.dirname(img_path_storage), exist_ok=True)
            img2.save(img_path_storage)

        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(112,96))
        img1 = TF.crop(img1, i, j, h, w)
        img2 = TF.crop(img2, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        
        img1 = 255*img1
        img2 = 255*img2
        sigma = pydlutil.wmad_estimator(img2)

        return img2, img1, class_id, sigma
    
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from skimage import img_as_float, img_as_ubyte, io


class MSRMyDemosaicDataset(Dataset):
    """Microsoft Demosaic dataset."""

    def __init__(self, root_dir, transform=None, pattern='bayer_rggb',
                 apply_bilinear=False, selection_file=''):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            selection_file (string) : file with image ids used for some purpose
                                      either train, validation or test
        """
        self.root_dir = root_dir
        self.transform = transform
        self.groundtruth = 'groundtruth/'
        self.input = 'input/'
        self.selection_file = selection_file
        selected_files = [s.strip() for s in open(selection_file).readlines()]
        # keep files according to selection_file
        self.listfiles_gt = []
        for file_ in selected_files:
            self.listfiles_gt.append(file_+'.png')
        self.listfiles_gt.sort()

        self.mask = None
        self.pattern = pattern
        self.apply_bilinear = apply_bilinear

    def __len__(self):
        return len(self.listfiles_gt)

    def compute_mask(self, pattern, im_shape):
        """
        Function compute_mask create a mask accordying to patter. The purpose
        of mask is to transform 2D image to 3D RGB.
        """
        # code from https://github.com/VLOGroup/joint-demosaicing-denoising-sem
        if pattern == 'bayer_rggb':
            r_mask = np.zeros(im_shape)
            r_mask[0::2, 0::2] = 1

            g_mask = np.zeros(im_shape)
            g_mask[::2, 1::2] = 1
            g_mask[1::2, ::2] = 1

            b_mask = np.zeros(im_shape)
            b_mask[1::2, 1::2] = 1
            mask = np.zeros(im_shape +(3,))
            mask[:, :, 0] = r_mask
            mask[:, :, 1] = g_mask
            mask[:, :, 2] = b_mask
        elif pattern == 'xtrans':
            g_mask = np.zeros((6,6))
            g_mask[0,0] = 1
            g_mask[0,2] = 1
            g_mask[0,3] = 1
            g_mask[0,5] = 1

            g_mask[1,1] = 1
            g_mask[1,4] = 1

            g_mask[2,0] = 1
            g_mask[2,2] = 1
            g_mask[2,3] = 1
            g_mask[2,5] = 1

            g_mask[3,0] = 1
            g_mask[3,2] = 1
            g_mask[3,3] = 1
            g_mask[3,5] = 1

            g_mask[4,1] = 1
            g_mask[4,4] = 1

            g_mask[5,0] = 1
            g_mask[5,2] = 1
            g_mask[5,3] = 1
            g_mask[5,5] = 1

            r_mask = np.zeros((6,6))
            r_mask[0,4] = 1
            r_mask[1,0] = 1
            r_mask[1,2] = 1
            r_mask[2,4] = 1
            r_mask[3,1] = 1
            r_mask[4,3] = 1
            r_mask[4,5] = 1
            r_mask[5,1] = 1

            b_mask = np.zeros((6,6))
            b_mask[0,1] = 1
            b_mask[1,3] = 1
            b_mask[1,5] = 1
            b_mask[2,1] = 1
            b_mask[3,4] = 1
            b_mask[4,0] = 1
            b_mask[4,2] = 1
            b_mask[5,4] = 1

            mask = np.dstack((r_mask,g_mask,b_mask))

            h, w = im_shape
            nh = np.ceil(h*1.0/6)
            nw = np.ceil(w*1.0/6)
            mask = np.tile(mask,(int(nh), int(nw),1))
            mask = mask[:h, :w,:]
        else:
            raise NotImplementedError('Only bayer_rggb is implemented')


        return mask

    def preprocess(self, pattern, img):
        """
        bilinear interpolation for bayer_rggb images
        """
        # code from https://github.com/VLOGroup/joint-demosaicing-denoising-sem
        if pattern == 'bayer_rggb':
            convertedImage = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB_EA)
            return convertedImage

        else:
            raise NotImplementedError('Preprocessing is implemented only for bayer_rggb')

    def __getitem__(self, idx):
        img_name_gt = os.path.join(self.root_dir,  self.groundtruth,
                                   self.listfiles_gt[idx])
        img_name_input = os.path.join(self.root_dir,  self.input,
                                      self.listfiles_gt[idx])

        image_gt = cv2.imread(img_name_gt)
        b, g, r = cv2.split(image_gt)       # get b,g,r
        image_gt = cv2.merge([r, g, b])     # switch it to rgb

        image_input = io.imread(img_name_input, )

        # perform mask computation
        mask = self.compute_mask(self.pattern, image_input.shape)
        mask = mask.astype(np.int32)
        image_mosaic = np.zeros(image_gt.shape).astype(np.int32)

        image_mosaic[:, :, 0] = mask[..., 0] * image_input
        image_mosaic[:, :, 1] = mask[..., 1] * image_input
        image_mosaic[:, :, 2] = mask[..., 2] * image_input
        #print(image_mosaic.dtype)
        image_input = np.sum(image_mosaic, axis=2, dtype='uint16')
        # perform bilinear interpolation for bayer_rggb images
        if self.apply_bilinear:
            image_mosaic = self.preprocess(self.pattern, image_input)

        image_gt = img_as_ubyte(image_gt)
        image_input = image_mosaic.astype(np.float32)/65535*255
        #assert image_gt.dtype == 'float64'
        #assert image_input.dtype == 'float64'
        bayer = image_input
        gt = image_gt
        if self.transform:
            bayer = self.transform(bayer)
            gt = self.transform(gt)
#         print("bayer", bayer.shape, "gt", gt.shape)

        return bayer, 255*gt
#         sample = {'image_gt': image_gt,
#                   'image_input': image_input,
#                   'filename': self.listfiles_gt[idx],
#                   'mask':mask}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample

    
class DemosaicDataset(torch.utils.data.Dataset):
    """Demosaic dataset."""

    def __init__(self, root_dir, transform=None, pattern='bayer_rggb', gt_dir_name="gt", bayer_dir_name="noised_bayer"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            selection_file (string) : file with image ids used for some purpose
                                      either train, validation or test
        """
        self.root_dir = root_dir
        self.transform = transform
        self.groundtruth = os.path.join(self.root_dir, gt_dir_name)
        self.bayer = os.path.join(self.root_dir, bayer_dir_name)
        
        self.listfiles = []
        classes_folders = sorted(os.listdir(self.groundtruth))
        self.classes = dict(zip(classes_folders, range(len(classes_folders))))

        for root, _, fnames in os.walk(self.groundtruth):
            for f in fnames:
                self.listfiles.append(os.path.join(os.path.split(root)[1], f))
        self.listfiles = sorted(self.listfiles)
        
    def __len__(self):
        return len(self.listfiles)

    def __getitem__(self, idx):
        label = self.classes[os.path.split(self.listfiles[idx])[0]]

        img_name_gt = os.path.join(self.groundtruth, self.listfiles[idx])
        img_name_bayer = os.path.join(self.bayer, self.listfiles[idx])
        
        bayer = Image.open(img_name_bayer)
        gt = Image.open(img_name_gt)
        
        if random.random() > 0.5:
            bayer = TF.hflip(bayer)
            gt = TF.hflip(gt)
        
#         bayer = TF.to_tensor(bayer)
#         gt = TF.to_tensor(gt)
        if self.transform:
            bayer = self.transform(bayer)
            gt = self.transform(gt)

        return 255*bayer, 255*gt, label