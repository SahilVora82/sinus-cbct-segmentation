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

# Normalize the scan using mean and standard deviation from the training set
TRAINING_MEAN = -34.69477476229485
TRAINING_STD = 244.6038824012994


def normalize():
    def execute(tensors: List[torch.Tensor]):
        with torch.no_grad():
            first = tensors[0]

            tensors[0] = (
                first - TRAINING_MEAN
            ) / TRAINING_STD

        return tensors

    return execute

def randomFlip(flip_dimensions: List[int], probability: float):
    def execute(tensors: List[torch.Tensor]):
        if(torch.rand([1]).item() < probability):
            return [ torch.flip(tensor, flip_dimensions) for tensor in tensors ]
        else:
            return tensors
    return execute


class EdgeBoostConfig():
    def __init__(self, region_size: int, boost_factor: int):
        self.region_size = region_size
        self.boost_factor = boost_factor
    def equals(self, other):
        return self.region_size == other.region_size and self.boost_factor == other.boost_factor

# Boosts the loss near the edges of the segments and in the positive segment.
# Returns the weight tensor to be applied on the loss.
# https://www.desmos.com/calculator/u6hoqlbtot
def edge_boost(segment: torch.Tensor, edge_boost_config: EdgeBoostConfig) -> torch.Tensor:
    region_size = edge_boost_config.region_size
    boost_factor = edge_boost_config.boost_factor
    with torch.no_grad():
        counting_pool = nn.AvgPool3d(region_size, 1, region_size//2, divisor_override=1)
        count = counting_pool(segment)
        _, unique_counts = torch.unique(segment, return_counts=True)
        neg_pos_ratio = unique_counts[0] / unique_counts[1]
        return edge_boost_function(count, region_size, neg_pos_ratio, boost_factor=boost_factor, dimensions=3, bias=0.6)

# Region size = Size of region of each dimension
# Negative positive ratio = Number of negative samples / number of positive samples
# bias = Where should max boost be applied (approximately)
#   0 = max boost applied on full negative regions
#   0.5 = max boost applied on half negative, half positive regions
#   1 = max boost applied on full positive regions
# dimensions = Number of dimensions of region
# boost_factor = Max boost amount. Will be multiplied by roughly the neg_pos_ratio
def edge_boost_function(tensor: torch.Tensor, region_size: int, neg_pos_ratio: float, boost_factor, dimensions, bias: float = 0.5) -> torch.Tensor:
    neg_pos_difference = neg_pos_ratio - 1
    half_volume = region_size**dimensions * 0.5
    max_boost_volume = region_size**dimensions * bias
    boost_range_factor = region_size**(region_size/(2 * dimensions))

    default_weight = neg_pos_difference * torch.reciprocal(1 + torch.pow(1 + 1/dimensions, -tensor + half_volume)) + 1

    edge_boost = boost_range_factor * boost_factor * neg_pos_difference * torch.reciprocal(torch.square(tensor - max_boost_volume) + boost_range_factor)
    edge_boost = edge_boost - (boost_range_factor * boost_factor * neg_pos_difference / (max_boost_volume**2 + boost_range_factor))
    return default_weight + edge_boost

class ScanDataset(Dataset):
    def __init__(self, data_directory, target_directory, target_suffix, edge_boost_config: EdgeBoostConfig | None, transformations: List[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = [], edge_boost_save_directory=None):
        data_directory = Path(data_directory)
        target_directory = Path(target_directory)
        self.data_directory = data_directory
        self.target_directory = target_directory

        self.target_suffix = target_suffix
        self.transformations = transformations
        
        self.counting_pool = nn.AvgPool3d(1, divisor_override=1)

        if(edge_boost_save_directory == None):
            edge_boost_save_directory = Path.joinpath(target_directory, "edge_boost_tensors")

        config_path = Path.joinpath(edge_boost_save_directory, "config.pt")
        # Regenerate edge boost tensors every time.
        if edge_boost_save_directory.exists():
            rmtree(edge_boost_save_directory)

        makedirs(edge_boost_save_directory, exist_ok=True)

        edge_boost_config_dict = {
            "edge_boost_config": edge_boost_config
        }
        torch.save(edge_boost_config_dict, config_path)
                
        self.edge_boost_save_directory = edge_boost_save_directory
        self.edge_boost_config = edge_boost_config
        
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
        edge_boost_tensor_path = Path.joinpath(self.edge_boost_save_directory, selected_id + self.target_suffix + ".pt")
        
        data, _ = nrrd.read(data_path)
        target, _ = nrrd.read(target_path)
        
        data = torch.from_numpy(data.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        
        # add channel dimension
        data.unsqueeze_(0)
        target.unsqueeze_(0)
        
        if(False == path.isfile(edge_boost_tensor_path)):
            if(self.edge_boost_config == None):
                loss_weights = torch.ones_like(target)
            else:
                loss_weights = edge_boost(target, self.edge_boost_config)
                torch.save(loss_weights, edge_boost_tensor_path)
        else:
            loss_weights: torch.Tensor = torch.load(edge_boost_tensor_path)
        
        for transformation in self.transformations:
            data, target, loss_weights = transformation([data, target, loss_weights])
        
        return data, target, loss_weights

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
        data, data_header = nrrd.read(data_path)
        data = torch.from_numpy(data.astype(np.float32))
        data.unsqueeze_(0)
        for transformation in self.transformations:
            [ data ] = transformation([data])
        return data, data_header, selected_id