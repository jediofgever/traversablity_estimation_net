import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_off
import open3d as o3d
import numpy as np
from torch_geometric.data import Data
import random

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

    
class TerrainDataset(InMemoryDataset):

    def __init__(self, root: str, 
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return '2d_circle'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']
    
    def process(self):
        print('Processing...')
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def extract_label(self, filename):
        # extract the label from the filename
        # delete last 4 characters (.pcd)
        filename = filename[:-4]
        # reverse the string
        filename = filename[::-1]
        # slice the string until the first underscore
        delimiter = "_"
        trimmed_string = filename.split(delimiter)[0]
        trimmed_string = trimmed_string[::-1]
        # convert to int 
        label = int(trimmed_string)
        # this label is a 0-1.0 float, divide by 100
        label = label / 100.0
        # force the label to be in the range of 0-1.0
        if label > 1.0:
            label = 1.0
        if label < 0.0:
            label = 0.0    
        return label   
    
    def process_set(self, dataset: str):
        
        aug_noise_rate = 0.5
        aug_downsample_rate = 0.5
        
        # get all the filenames in the dataset
        categories = glob.glob(osp.join(self.raw_dir, dataset))
        files = os.listdir(categories[0])
        data_list = []
        for i in range(len(files)):
            files[i] = osp.join(categories[0], files[i])
            # this is a pcd file, read with open3d
            o3d_cloud = o3d.io.read_point_cloud(files[i])
            
            label = self.extract_label(files[i])             
            points = np.asarray(o3d_cloud.points).astype(np.float32)
            points = pc_normalize(points)
            points = torch.tensor(points)
            
            #points = points - points.mean(dim=0, keepdim=True)
            label = torch.tensor(np.asarray([label]).astype(np.float32))
            data = Data(pos=points, y=torch.tensor([label]), face=torch.tensor([]))
            data_list.append(data)
        
        aug_samples = 0
        print("Raw data list size with augmented ops: ", len(data_list))    
        if aug_noise_rate > 0.0:
            augmented_data_list = self.aug_noise(files, categories, aug_noise_rate)
            print("Choosed to augment with noise rate of: ", aug_noise_rate)
            data_list = data_list + augmented_data_list
            aug_samples = aug_samples + len(augmented_data_list)
        
        if aug_downsample_rate > 0.0:
            augmented_data_list = self.aug_downsample(files, categories, aug_downsample_rate)
            print("Choosed to augment with downsample rate of: ", aug_downsample_rate)
            data_list = data_list + augmented_data_list
            aug_samples = aug_samples + len(augmented_data_list)
        
        print("Total data list size with augmented ops: ", len(data_list))   
        print("Added ", aug_samples, " augmented samples")
            
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
            
        return self.collate(data_list)

    def aug_downsample(self, files, categories, aug_downsample_rate):
        # Now select a random files from the list with self.aug_downsample rate
        n = aug_downsample_rate * len(files)
        # printing n elements from list
        selected_files = random.choices(files, k=int(n))
                
        aug_data_list = []
        
        for i in range(len(selected_files)):
            selected_files[i] = osp.join(categories[0], selected_files[i])
            # this is a pcd file, read with open3d
            o3d_cloud = o3d.io.read_point_cloud(selected_files[i])
            #Downsample the point cloud with random voxel size
            voxel_size = np.random.uniform(0.05, 0.25)
            o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size)
            label = self.extract_label(selected_files[i])             
            points = np.asarray(o3d_cloud.points).astype(np.float32)
            points = pc_normalize(points)
            points = torch.tensor(points)
            points = points - points.mean(dim=0, keepdim=True)
            label = torch.tensor(np.asarray([label]).astype(np.float32))
            data = Data(pos=points, y=torch.tensor([label]), face=torch.tensor([]))
            aug_data_list.append(data)
        
        return aug_data_list    
        
        
    def aug_noise(self, files, categories, aug_noise_rate):
        # Now select a random files from the list with self.aug_downsample rate
        n = aug_noise_rate * len(files)
        # printing n elements from list
        selected_files = random.choices(files, k=int(n))
                
        aug_data_list = []
        
        for i in range(len(selected_files)):
            selected_files[i] = osp.join(categories[0], selected_files[i])
            # this is a pcd file, read with open3d
            o3d_cloud = o3d.io.read_point_cloud(selected_files[i])
            #Add noise the point cloud  
            points = np.asarray(o3d_cloud.points).astype(np.float32)
            noise = np.random.normal(0.01, 0.04, points.shape)
            points = points + noise
            o3d_cloud.points = o3d.utility.Vector3dVector(points)
            
            label = self.extract_label(selected_files[i])             
            points = np.asarray(o3d_cloud.points).astype(np.float32)
            points = pc_normalize(points)
            points = torch.tensor(points)
            points = points - points.mean(dim=0, keepdim=True)
            label = torch.tensor(np.asarray([label]).astype(np.float32))
            data = Data(pos=points, y=torch.tensor([label]), face=torch.tensor([]))
            aug_data_list.append(data)
        
        return aug_data_list       