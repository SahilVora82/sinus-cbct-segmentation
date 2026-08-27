import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import nrrd

from pathlib import Path
from os import listdir, path, makedirs
from shutil import rmtree

from typing import List, Callable


# Some basic preprocessing and data augmentation functions

# Normalize only the first tensor to range 0 -> 1
def normalize():
    def execute(tensors: List[torch.Tensor]):
        with torch.no_grad():
            first = tensors[0]
            mean = torch.mean(first)
            std = torch.std(first)
            tensors[0] = (first - mean) / std
        return tensors
    return execute

def randomFlip(flip_dimensions: List[int], probability: float):
    def execute(tensors: List[torch.Tensor]):
        if(torch.rand([1]).item() < probability):
            return [ torch.flip(tensor, flip_dimensions) for tensor in tensors ]
        else:
            return tensors
    return execute

class ScanDataset(Dataset):
    def __init__(self, data_directory, target_directory, target_suffix, transformations: List[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = []):
        data_directory = Path(data_directory)
        target_directory = Path(target_directory)
        self.data_directory = data_directory
        self.target_directory = target_directory

        self.target_suffix = target_suffix
        self.transformations = transformations
        
        self.counting_pool = nn.AvgPool3d(1, divisor_override=1)

        self.ids = []
        for file_name in listdir(data_directory):
            if path.isfile(Path.joinpath(data_directory, file_name)) and not file_name.startswith("."):
                self.ids.append(file_name.split(".")[0])
        
        print(f"{len(self.ids)} scans found at {data_directory}.")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        selected_id = self.ids[index]
        data_path = Path.joinpath(self.data_directory, selected_id + ".nrrd")
        target_path = Path.joinpath(self.target_directory, selected_id + self.target_suffix + ".nrrd")
        
        data, _ = nrrd.read(str(data_path))
        target, _ = nrrd.read(str(target_path))
        
        data = torch.from_numpy(data.astype(np.float32))
        target = torch.from_numpy(target).to(torch.long)
        
        # add channel dimension
        data.unsqueeze_(0)
        # target unnecessary because class values in data
        
        for transformation in self.transformations:
            data, target = transformation([data, target])
        
        return data, target

# Returns data and data_header instead of data and target
class SingleScanDataset(Dataset):
    def __init__(self, data_directory, transformations: List[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = []):
        data_directory = Path(data_directory)
        self.data_directory = data_directory
        self.transformations = transformations
        self.ids = []
        for file_name in listdir(data_directory):
            if path.isfile(Path.joinpath(data_directory, file_name)) and not file_name.startswith("."):
                self.ids.append(file_name.split(".")[0])
        print(f"{len(self.ids)} scans found at {data_directory}.")
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        selected_id = self.ids[index]
        data_path = Path.joinpath(self.data_directory, selected_id + ".nrrd")
        data, data_header = nrrd.read(str(data_path))
        data = torch.from_numpy(data.astype(np.float32))
        data.unsqueeze_(0)
        for transformation in self.transformations:
            [ data ] = transformation([data])
        return data, data_header, selected_id