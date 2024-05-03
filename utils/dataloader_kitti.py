from logging import root
import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
import scipy.io as sio
import sys
from io_util import read_pcd, read_mat
from data_util import check_degree, resample_pcd
import cv2

class TestDataLoader(Dataset):
    def __init__(self,filepath,data_path,status,pc_input_num=2048,category='all'):
        super(TestDataLoader,self).__init__()
        """
        Args:
        filepath: list of dataset
        data_path: root path of training and test data
        status: 'valid' or 'test'

        """
        self.cat_map = {
            'plane':'02691156',
            'cabinet':'02933112',
            'car':'02958343',
            'chair':'03001627',
            'lamp':'03636649',
            'couch':'04256520',
            'table':'04379243',
            'watercraft':'04530566'
        }
        self.pc_input_num = pc_input_num
        self.filelist = os.listdir(os.path.join(data_path, 'image'))
        self.cat = []
        self.key = []    
        
        self.incomplete_path = os.path.join(data_path, 'train_part_newscale_whl') 
        self.rendering_path = os.path.join(data_path,'image')
        self.silh_path = os.path.join(data_path, 'mask')
        self.bound_path = os.path.join(data_path, 'contours_fps')
        self.param_path = os.path.join(data_path, 'train_param_newscale_whl')

        for key in self.filelist:
            self.key.append(key[:-4])

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def __getitem__(self, idx):
        view_set = np.empty((1,3, 224, 224))
        param_set = np.empty((1,4,4))
        inv_set = np.empty((1,4,4)) # inverse param
        silh_set = np.empty((1,224, 224))
        bound_set = np.empty((1,240, 2)) # 240 points for each contour
        key = self.key[idx].strip()
        pc_part_path = os.path.join(self.incomplete_path,key+'.npy')
        view_path = os.path.join(self.rendering_path,key+'.jpg') # e.g:categ/obj/
        silh_path = os.path.join(self.silh_path, key+'.jpg')
        param_path = os.path.join(self.param_path, key+'.mat')
        bound_path = os.path.join(self.bound_path, key+'.mat')

        views = self.transform(Image.open(view_path)) # [C, H, W]
        views = views[:3,:,:]
        view_set[0] = views
        param_set[0] = sio.loadmat(param_path)['proj_mat']
        inv_set[0] = sio.loadmat(param_path)['proj_inv']
        silh_set[0] = cv2.imread(silh_path, flags=-1)
        bound_set[0] = sio.loadmat(bound_path)['points']
        ## load partial points
        pc_part = np.load(pc_part_path)
        if pc_part.shape[0]!=self.pc_input_num:
            pc_part = resample_pcd(pc_part, self.pc_input_num)

        return torch.from_numpy(view_set).float(), torch.from_numpy(pc_part).float(), \
            torch.from_numpy(param_set).float(),\
            torch.from_numpy(inv_set).float(), torch.from_numpy(silh_set).float(),\
            torch.from_numpy(bound_set).float(), key



    def __len__(self):
        return len(self.key)

