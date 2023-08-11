using FileIO
using PyCall

py"""
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def show_batch(batch):
    batch = torch.from_numpy(batch).float()
    for i, img in enumerate(batch):
        # Plot the fake images 
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Real Image")
        plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True),(1,2,0)))
        plt.show()
        print(f"Image [{i+1}/{batch.shape[0]}]")
"""

# load batches
file_name_data = "CelebA-HQ_preprocessed.jld2"
data = load(file_name_data, "data") # a vector of batches
# print type of data 
println("Type of data: $(typeof(data))")
# print the number of batches
println("Number of batches: $(length(data))")
# select one batch
batch_index = 1
batch = data[batch_index]
# print size of one batch
println("Size of one batch: $(size(batch))")
# reshape from WHCN to NCHW for pytorch
batch = permutedims(batch, (4, 3, 2, 1))
py"show_batch"(batch)