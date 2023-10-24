#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
import math
import json
from PIL import Image
from io import BytesIO
import base64
import cv2

def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a Gaussian distribution with standard deviation = sigma
    and sum of all elements = 1.

    Length of the list = window_size
    """
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a Gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

    # Converting to 2D
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images)

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    if window is None:
        real_size = min(window_size, height, width)  # window should be at least 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret

def tensorify(x):
    return torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)

def process_image(data):
    data = json.loads(data)
    user_image_base64 = data['userImage']
    dataset_image_path = data['datasetImage']

    # Decode the base64 user image and save it as a file
    user_image_bytes = base64.b64decode(user_image_base64)
    user_image = Image.open(BytesIO(user_image_bytes))
    user_image = user_image.resize((480, 640))
    
    # Load the reference image
    dataset_image = Image.open(dataset_image_path)
    dataset_image = dataset_image.resize((480, 640))

    # Calculate SSIM score
    _user_image = tensorify(np.asarray(user_image))
    _dataset_image = tensorify(np.asarray(dataset_image))
    ssim_score = ssim(_user_image, _dataset_image, val_range=255)

    return ssim_score.item()

if __name__ == "__main__":
    import sys
    data = json.loads(sys.argv[1])
    # print(data)
    result = process_image(data)
    print(result * 100)
