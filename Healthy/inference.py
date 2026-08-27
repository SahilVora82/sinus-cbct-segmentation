import os
from pathlib import Path

import nrrd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import normalize, SingleScanDataset

from config import model, device
from config import inference_checkpoint_directory
from config import inference_checkpoint_file
from config import inference_scans_directory
from config import inference_output_directory
from config import inference_segments_suffix


os.makedirs(inference_output_directory, exist_ok=True)


transformations = [
    normalize()
]


inference_dataset = SingleScanDataset(
    inference_scans_directory,
    transformations=transformations
)


inference_dataloader = DataLoader(
    inference_dataset,
    batch_size=1,
    collate_fn=lambda batch: batch[0]
)


if inference_checkpoint_file is None:
    raise ValueError(
        "Set inference_checkpoint_file in config.py before running inference."
    )


checkpoint_path = (
    Path(inference_checkpoint_directory)
    / inference_checkpoint_file
)


state = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(state["model"])
model.eval()


with torch.no_grad():
    for sample, sample_header, sample_id in inference_dataloader:
        sample = sample.unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            prediction = model(sample)

        prediction = prediction.squeeze(0).squeeze(0)

        prediction = (
            torch.sigmoid(prediction) >= 0.5
        ).to(torch.uint16)

        prediction_numpy = prediction.cpu().numpy().astype(np.uint16)

        segment_file_name = (
            Path(inference_output_directory)
            / f"{sample_id}{inference_segments_suffix}.nrrd"
        )

        nrrd.write(
            str(segment_file_name),
            prediction_numpy,
            sample_header
        )

        print(f"Saved prediction: {segment_file_name}")