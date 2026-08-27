import torch
import numpy as np
import nrrd
import matplotlib.pyplot as plot

ct_file_path = "./data/scan/CBCT1_cropped.nrrd"
ct_data, ct_header = nrrd.read(ct_file_path)

seg_file_path = "./data/segment/CBCT1_cropped_GT.nrrd"
seg_data, seg_header = nrrd.read(seg_file_path)

import matplotlib.animation as animation

figure, axes = plot.subplots()

ct_data = torch.tensor(ct_data.astype(np.int64))
ct_data = torch.flip(ct_data, [0])
ct_data = ct_data.numpy()

seg_data = torch.tensor(seg_data.astype(np.int64))
seg_data = torch.flip(seg_data, [0])
seg_data = seg_data.numpy()

images = []
max_value = np.max(ct_data)
for i in range(224):
    image = axes.imshow(ct_data[:, i, :] / max_value + seg_data[:, i, :], cmap="gray", vmin=0, vmax=2, origin="lower", animated=True)
    images.append([image])

video = animation.ArtistAnimation(figure, images, interval=25, blit=True)
# video.save("./animations/scan.mp4")

plot.show()