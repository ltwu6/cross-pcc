import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import scipy.io as sio
from data_util import resample_pcd
import cv2


class TrainDataLoader(Dataset):
    def __init__(self,filepath,data_path, pc_input_num=2048,category='all'):
        super(TrainDataLoader,self).__init__()
        """
        Args:
        filepath: list of dataset
        data_path: root path of training and test data
        status: 'train', 'valid' or 'test'

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
        self.data_path = data_path
        self.filepath = os.path.join(filepath, 'train.txt')
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        
        with open(self.filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line.strip())
                line = f.readline()
        
        self.incomplete_path = os.path.join(self.data_path,'train/part') 
        self.rendering_path = os.path.join(self.data_path, 'all/image')
        self.point2d_path = os.path.join(self.data_path, 'all/point2d_thred005')
        self.silh_path = os.path.join(self.data_path, 'all/mask_thred005')
        self.bound_path = os.path.join(self.data_path, 'all/contours_thred005_fps')
        self.param_path = os.path.join(self.data_path, 'all/revise_mat')
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split('/')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ]) 

        print(f'train data num: {len(self.key)}')


    def __getitem__(self, idx):
        view_set = np.empty((8, 3, 224, 224))
        param_set = np.empty((8, 4,4))
        inv_set = np.empty((8, 4,4)) # inverse param
        point2d_set = np.empty((8,4096,2))
        silh_set = np.empty((8, 224, 224))
        bound_set = np.empty((8, 240, 2)) # 240 points for each contour
        key = self.key[idx].strip()
        pc_part_path = os.path.join(self.incomplete_path,key)+'.npy'
        view_path = os.path.join(self.rendering_path,key[:-1]) # e.g:categ/obj/
        point2d_path = os.path.join(self.point2d_path,key[:-1]) # e.g:categ/obj/
        silh_path = os.path.join(self.silh_path, key[:-1])
        param_path = os.path.join(self.param_path, key[:-1])
        bound_path = os.path.join(self.bound_path, key[:-1])
        view_metadata = np.loadtxt(view_path+'new_rendering_metadata.txt')
        for i in range(8):
            views = self.transform(Image.open(view_path+str(i).rjust(2,'0')+'.png')) # [C, H, W]
            views = views[:3,:,:]
            view_set[i] = views
            param_set[i] = sio.loadmat(param_path+'camera_'+str(i)+'.mat')['proj_mat']
            inv_set[i] = sio.loadmat(param_path+'camera_'+str(i)+'.mat')['proj_inv']
            point2d_set[i] = sio.loadmat(point2d_path+str(i)+'.mat')['points']
            silh_set[i] = cv2.imread(silh_path+str(i).rjust(2,'0')+'.png', flags=-1)
            bound_set[i] = sio.loadmat(bound_path+str(i)+'.mat')['points']
        pc_part = np.load(pc_part_path)
        if pc_part.shape[0]!=self.pc_input_num:
            pc_part = resample_pcd(pc_part, self.pc_input_num)
        return torch.from_numpy(view_set).float(), torch.from_numpy(pc_part).float(),\
               torch.from_numpy(param_set).float(), torch.from_numpy(point2d_set/224.0).float(),\
               torch.from_numpy(silh_set/255.0).float(), torch.from_numpy(inv_set).float(),\
               torch.from_numpy(bound_set).float(), key

    def __len__(self):
        return len(self.key)

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
        self.status = status
        self.data_path = data_path
        self.filepath = os.path.join(filepath, self.status+'.txt')
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category

        with open(self.filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line.strip())
                line = f.readline()
        
        self.incomplete_path = os.path.join(self.data_path, self.status,'part') 
        self.gt_path = os.path.join(self.data_path, self.status, 'gt')
        self.rendering_path = os.path.join(self.data_path, 'all/image')
        self.silh_path = os.path.join(self.data_path, 'all/mask_thred005')
        self.bound_path = os.path.join(self.data_path, 'all/contours_thred005_fps')
        self.param_path = os.path.join(self.data_path, 'all/revise_mat')

        # should modify
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split('/')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print(f'{self.status} data num: {len(self.key)}')


    def __getitem__(self, idx):
        view_set = np.empty((8, 3, 224, 224))
        param_set = np.empty((8, 4,4))
        inv_set = np.empty((8, 4,4)) # inverse param
        silh_set = np.empty((8, 224, 224))
        bound_set = np.empty((8, 240, 2)) # 240 points for each contour
        key = self.key[idx].strip()
        pc_part_path = os.path.join(self.incomplete_path,key+'.npy')
        pc_path = os.path.join(self.gt_path,key[:-1]+'0.npy')
        view_path = os.path.join(self.rendering_path,key[:-1]) # e.g:categ/obj/
        silh_path = os.path.join(self.silh_path, key[:-1])
        param_path = os.path.join(self.param_path, key[:-1])
        bound_path = os.path.join(self.bound_path, key[:-1])

        # load view and params
        view_metadata = np.loadtxt(view_path+'new_rendering_metadata.txt')
        for i in range(8):
            views = self.transform(Image.open(view_path+str(i).rjust(2,'0')+'.png')) # [C, H, W]
            views = views[:3,:,:]
            view_set[i] = views
            param_set[i] = sio.loadmat(param_path+'camera_'+str(i)+'.mat')['proj_mat']
            inv_set[i] = sio.loadmat(param_path+'camera_'+str(i)+'.mat')['proj_inv']
            silh_set[i] = cv2.imread(silh_path+str(i).rjust(2,'0')+'.png', flags=-1)
            bound_set[i] = sio.loadmat(bound_path+str(i)+'.mat')['points']
        pc = np.load(pc_path)
        pc_part = np.load(pc_part_path)
        if pc_part.shape[0]!=self.pc_input_num:
            pc_part = resample_pcd(pc_part, self.pc_input_num)

        return torch.from_numpy(view_set).float(), torch.from_numpy(pc_part).float(), \
            torch.from_numpy(pc).float(), torch.from_numpy(param_set).float(),\
            torch.from_numpy(inv_set).float(), torch.from_numpy(silh_set).float(),\
            torch.from_numpy(bound_set).float(), key



    def __len__(self):
        return len(self.key)

