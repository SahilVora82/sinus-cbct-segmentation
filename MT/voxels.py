import torch
import numpy as np
import nrrd
import matplotlib.pyplot as plot

ct_file_path = "./data/scan/CBCT1_cropped.nrrd"
ct_data, ct_header = nrrd.read(ct_file_path)

seg_file_path = "./data/segment/CBCT1_cropped_GT.nrrd"
seg_data, seg_header = nrrd.read(seg_file_path)

ct_data, seg_data = ct_data.astype(np.int64), seg_data.astype(np.int64)

# Input is 3d tensor
def get_nonempty_and_facecolors(tensor, threshold):
    avg_pool = torch.nn.AvgPool3d(4)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = avg_pool(tensor)
    tensor = tensor[0, 0]

    # Threshold for empty, no plotting
    nonempty_locations = tensor >= threshold
    plot_alphas = torch.where(nonempty_locations, (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor)), 0)
    plot_alphas = torch.unsqueeze(plot_alphas, -1)
    plot_rgb = torch.ones((*tensor.shape, 3))
    plot_rgba = torch.cat([plot_rgb, plot_alphas], dim=-1)
    return nonempty_locations, plot_rgba

ct_nonempty, ct_face_colors = get_nonempty_and_facecolors(torch.tensor(ct_data, dtype=torch.float), np.quantile(ct_data, 0.9))
seg_nonempty, seg_face_colors = get_nonempty_and_facecolors(torch.tensor(seg_data, dtype=torch.float), np.mean(seg_data))

figure, axes = plot.subplots(subplot_kw={"projection": "3d"})

axes.voxels(ct_nonempty.numpy(), facecolors=ct_face_colors.numpy())
axes.voxels(seg_nonempty.numpy(), facecolors="tab:red", alpha=0.9)
axes.view_init(roll=90)

plot.show()