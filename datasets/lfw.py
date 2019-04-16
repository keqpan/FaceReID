import torch
import torchvision.datasets as datasets
import os
import numpy as np

class LFWDataset(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dir,pairs_path, transform=None, file_ext="jpg"):

        super(LFWDataset, self).__init__(dir,transform)

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_lfw_paths(dir, file_ext)

    def read_lfw_pairs(self,pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self,lfw_dir,file_ext="jpg"):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
        #for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

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

        (path_1,path_2,issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame


    def __len__(self):
        return len(self.validation_images)


from PIL import Image
import os
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class QMULDataset(torch.utils.data.Dataset):
    '''
    '''
    def __init__(self, dir, pairs_path, transform=None, file_ext="png"):

        super(QMULDataset, self).__init__()

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_qmul_paths(dir, file_ext)
        self.root = dir
        self.transform = transform
        self.loader = pil_loader

    def read_qmul_pairs(self, pairs_filename):
        pairs = np.load(pairs_filename)
        return pairs

    def get_qmul_paths(self, qmul_dir, file_ext="png"):

        pairs = self.read_qmul_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        fold_size = 532
        n_folds = 10
        issame_list = np.tile(np.hstack((np.zeros(fold_size), np.ones(fold_size))).astype(np.bool), n_folds).reshape(-1,1)
        return np.hstack((pairs, issame_list))

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

        path_1, path_2, issame = self.validation_images[index]
        path_1 = os.path.splitext(path_1)[0]+".png"
        path_2 = os.path.splitext(path_2)[0]+".png"
        path_1 = os.path.join(self.root, path_1)
        path_2 = os.path.join(self.root, path_2)
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame


    def __len__(self):
        return len(self.validation_images)