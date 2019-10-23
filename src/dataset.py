import numpy as np
import os, cv2

import torch
import torch.utils.data as D

class Identity():
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        pass
    
    def __repr__(self):
        format_string = self.__class__.__name__ 
        return format_string
    
    def __call__(self, input):
        return input

class FashionDataset(D.Dataset):
    
    def __init__(self, filename, masks=None, path='input', mode='train',
                             transform=None, size=(512,512)):
        r""" Dataset for kaggle competition iMaterialist.
        https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6

        Args:
            filename: list of image files
            masks: dictionary, which contained structured masks by filename
            path: path to images
            mode: 'train' or 'test mode, if mode test returned image and filename
            transform: transform and augmentation function for image
            seze: all image will be resize to this (height,width)
        """
        if mode != 'test': assert masks is not None, 'For train mode need masks!'
            
        self.filename = filename    
        self.transform = transform if transform else Identity()
        self.path = path
        self.mode = mode
        self.masks = masks
        self.len = len(self.filename)
        self.size = size

    def image(self, index):
        """ Load image """
        
        filename = self.filename[index]
        image = cv2.imread(os.path.join(self.path, filename), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, cv2.INTER_AREA)
        return image
    
    @staticmethod
    def rle_decode(mask_rle, shape=(256, 256)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background

        '''
        
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape, order='F')

    def mask(self, index):
        """ Decode rle mask and resize it """
        
        filename = self.filename[index]
        h, w = self.masks[filename]['Height'], self.masks[filename]['Width']
        mask = np.zeros((46, *self.size))
        for m in self.masks[filename]['mask']:
            image = self.rle_decode(m[1], (h, w))
            mask[m[0]] = cv2.resize(image, self.size, cv2.INTER_NEAREST)
        return mask
    
    def __getitem__(self, index):
        """ Return Image and Mask or Image if mode 'test' """
        
        image = self.image(index)
        image = self.transform(image)
        
        if self.mode == 'test':
            return image, self.filename[index]
        
        mask = self.mask(index)
        return image, mask
            
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len