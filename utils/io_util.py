'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import open3d as o3d
import scipy.io as sio
import torch
import cv2
from PIL import Image
from torchvision import utils as vutils


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def read_mat(filename):
    pc_mat = sio.loadmat(filename)
    return pc_mat['points']

def save_pcd(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def save_mat(filename, points):
    sio.savemat(filename, {'points': points})

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    :param input_tensor: 
    :param filename: 
    """
    img = input_tensor.detach().cpu().numpy()
    img = Image.fromarray(np.uint8(img))
    img.save(filename)









