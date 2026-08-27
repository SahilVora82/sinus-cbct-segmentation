import torch
from torch.utils.data import DataLoader
from dataset import normalize, SingleScanDataset

from config import model, device
from config import inference_checkpoint_directory, inference_checkpoint_file
from config import inference_scans_directory, inference_output_directory, inference_segments_suffix

import nrrd
import numpy as np

import os
from pathlib import Path
from typing import cast


os.makedirs(inference_output_directory, exist_ok=True)

transformations = [ normalize() ]
inference_dataset = SingleScanDataset(inference_scans_directory, transformations=transformations)
# Force single sample, collate problems when multiple, too lazy to introduce
# collate function
inference_dataloader = DataLoader(inference_dataset, batch_size=1, collate_fn=lambda batch: batch[0])

if(inference_checkpoint_file == None):
    print("Please specify a checkpoint file to load checkpoints from.")
    exit(1)

state = torch.load(Path.joinpath(inference_checkpoint_directory, inference_checkpoint_file))
model.load_state_dict(state["model"])


model.eval()
with torch.no_grad():
    # Single sample at a time
    for sample, sample_header, sample_id in inference_dataloader:
        sample = sample.unsqueeze(0).to(device) # Creates batch dimension
        prediction = cast(torch.Tensor, model(sample))
        prediction.squeeze_(dim=0) # Removes batch dimension
        prediction = torch.softmax(prediction, dim=0)
        prediction = torch.argmax(prediction, dim=0) # Will remove class dimension
        prediction = prediction.to(torch.int32)
        prediction = cast(np.ndarray, prediction.detach().cpu().resolve_conj().resolve_neg().numpy())
        prediction = prediction.astype(np.uint16) # Original segment datatype is uint16
        
        segment_file_name = Path.joinpath(inference_output_directory, sample_id + inference_segments_suffix + ".nrrd")
        nrrd.write(str(segment_file_name), prediction, sample_header)
