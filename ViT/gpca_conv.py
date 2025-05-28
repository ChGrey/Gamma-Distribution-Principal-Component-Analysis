import os.path
import torch.nn.functional as F
import torch
import scipy.io as scio
import math
import torch.nn as nn
import matplotlib.pyplot as plt

"""
embedding_input: torch.Size([BS, 3, 224, 224]) 
channel: original image channel*3
embedding_output: torch.Size([BS, 3, 224, 224]) 
channel: original image channel + principle component #1 + principle component #2
"""

# Put data on GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Conduct max-min scale for each channel of each image, respectively.
def max_min_scale(inputs):
    out_norm = torch.zeros(inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3),).to(device)
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            input_ij = inputs[i, j, :, :].to(device)
            input_ij_max = input_ij.max().to(device)
            input_ij_min = input_ij.min().to(device)
            input_normalized = (input_ij - input_ij_min) / (input_ij_max - input_ij_min).to(device)
            out_norm[i, j] = input_normalized
    return out_norm


def max_min(inputs):
    out_norm = torch.zeros(inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3),).to(device)
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            input_ij = inputs[i, j, :, :].to(device)
            input_ij_max = input_ij.max().to(device)
            input_ij_min = input_ij.min().to(device)
            input_normalized = 2*(input_ij - input_ij_min) / (input_ij_max - input_ij_min)-1
            out_norm[i, j] = input_normalized.to(device)
    return out_norm


# Input: GammaPCA_kernel.mat, Output: [kernel_number, kernel_channel, kernel_size, kernel_size]
def get_kernel(kernel_root, name):
    load_mat = scio.loadmat(os.path.join(kernel_root, name+'.mat'))
    kernel = load_mat['V0']

    # Preprocessing
    kernel = torch.FloatTensor(kernel).to(device)  # transform into tensor
    kernel = kernel.permute(1, 0)  # [kernel_size^2,kernel_number]->[kernel_number^2，kernel_size]
    ks = int(math.sqrt(kernel.size(1)))  # kernel_size
    kn = kernel.size(0)  # kernel_number
    # SAR images only have one channel, so GPCA kernel only has one channel as well.
    gpca_kernel = torch.zeros(kn, 1, ks, ks)
    for i in range(kn):
        ki = kernel[i]
        ki = ki.reshape(ks, ks).t()  # [kernel_size^2] -> reshape ->[kernel_size, kernel_size]
        ki = ki.expand(1, 1, ks, ks)  # add dimensions -> [1, 1, kernel_size, kernel_size]
        gpca_kernel[i, :, :, :] = ki  # stack kernels up -> [kernel_number, 1, kernel_size, kernel_size]
    gpca_kernel = torch.nn.Parameter(data=gpca_kernel, requires_grad=False).to(device)

    return gpca_kernel, ks, kn


def gpca_conv(imgs, kernel_dir):
    # parameters of GammaPCA kernel
    kernel_root = kernel_dir
    # kernel1, ks1, kn1 = get_kernel(kernel_root, 'ks7')
    # kernel2, ks2, kn2 = get_kernel(kernel_root, 'ks13')
    kernel3, ks3, kn3 = get_kernel(kernel_root, 'ks17')

    # Preprocess input images, imgs-[batch_size, channel, 224, 224]
    # The 3-channel SAR images are obtained by copying original gray SAR images for three times,
    # thus select only one channel for subsequent processing
    inputs = imgs[:, 0, :, :].unsqueeze(1).to(device)  # [BS, 1, 224, 224]

    # normalization
    # inputs = max_min_scale(inputs)

    # GPCA convolution
    # feature1 = F.conv2d(inputs, kernel1, padding=math.floor(ks1/2), stride=1).to(device)
    # feature2 = F.conv2d(inputs, kernel2, padding=math.floor(ks2/2), stride=1).to(device)
    feature3 = F.conv2d(inputs, kernel3, padding=math.floor(ks3/2), stride=1).to(device)
    # concatenate the original images and feature maps in channel-dimension to get 3-channel outputs
    outputs = torch.concat((inputs, feature3[:, 0, :, :].unsqueeze(1), feature3[:, 1, :, :].unsqueeze(1)), dim=1).to(device)

    # normalization
    out_norm = max_min(outputs)

    # other normalization methods
    # norm = nn.LayerNorm([3, 224, 224]).to(device)  # LayerNorm
    # norm = nn.InstanceNorm2d(3).to(device)  # InstanceNorm
    # norm = nn.BatchNorm2d(3).to(device)  # BatchNorm
    # out_norm = norm(outputs)

    # visualization
    # out11 = out_norm[0, 0, :, :].cpu().numpy()
    # plt.subplot(1, 3, 1)
    # plt.imshow(out11)
    # out12 = out_norm[0, 1, :, :].cpu().numpy()
    # plt.subplot(1, 3, 2)
    # plt.imshow(out12)
    # out13 = out_norm[0, 2, :, :].cpu().numpy()
    # plt.subplot(1, 3, 3)
    # plt.imshow(out13)
    # plt.show()

    # print("embedding：", outputs.shape)  # [BS, 3, 224, 224]
    return out_norm


