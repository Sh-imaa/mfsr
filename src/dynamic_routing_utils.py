import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
from skimage.util.shape import view_as_windows

def get_routing_clusters(c, limit=1e1):
    # get clusters
    c = np.array([x.item() for x in c])
    c_sorted = np.sort(-c)
    sorted_i = np.argsort(-np.array(c))
    ep = 1e-100
    c_sorted -= ep
    gaps = np.array([(c_sorted[i] - c_sorted[i + 1])/ c_sorted[i + 1] for i in range(len(c_sorted) - 1)])
    if gaps.max() > limit:
        # contains bad images
        for i, g in enumerate(gaps):
            if g > limit:
                good_images = sorted_i[:i + 1]
                clusters = np.zeros_like(sorted_i)
                clusters[good_images] = 1
                return sorted_i[i + 1:], good_images, clusters

    return [], sorted_i, np.ones_like(sorted_i)


def route(images, it=500):
    """
    Args:
        images: list of nxm tensors
        it: number of iterations of the routing

    Returns:
    list of weights for every input tensor
    """
    
    images_flatten = [image.flatten() for image in images]
    images_t = torch.stack(images_flatten)
    images_t = torch.nn.functional.normalize(images_t, dim=1)

    n = len(images)
    b = torch.tensor([1. for _ in range(n)])
    c = nn.Softmax()(b).view(-1, 1)

    for _ in range(it):
        s = torch.mul(c, images_t).sum(dim=0)
        s = torch.nn.functional.normalize(s, dim=0).view(-1, 1)
        b = b + torch.stack([torch.mm(images_t[i].view(1, -1), s) for i in range(n)]).squeeze()
        c = nn.Softmax()(b).view(-1, 1)
    return c.squeeze()

def dist(v1, v2, dim=0):
    "Get Euclidean distance"
    return torch.pow(v1 - v2, 2).sum(dim=dim)

def route_per_pixel(images, its=100):
    n, ch, w, h = images.shape
    b = torch.ones(n, w, h)
    c = nn.Softmax(dim=0)(b)
    for _ in range(its):
        c_ = c.unsqueeze(1).repeat((1, ch, 1, 1))
        # a weighted image
        s = torch.mul(c_, images).sum(dim=0)
        b_ = torch.stack([-dist(images[i], s) for i in range(n)])
        b += b_.squeeze()
        c = nn.Softmax(dim=0)(b)
    return c


def route_conv(images_t, f=20, k=3, it=100, step=1):
    padding = int(k // 2)
    images_t = images_t[:, :f, :, :]
    n, ch, w, h = images_t.shape
    images_t = view_as_windows(images_t , (n, ch, k, k), step=step)
    images_t = torch.from_numpy(np.moveaxis(images_t, 4, 2))
    _, _, _, w_, h_, _, _, _ = images_t.shape
    images_t = images_t.reshape(n, w_, h_, -1) 
    images_t = torch.nn.functional.normalize(images_t, dim=-1)
    b = torch.ones(n, w_, h_)
    c = nn.Softmax(dim=0)(b)
    for _ in range(it):
        c_ = c.unsqueeze(3).repeat((1, 1, 1, ch * k * k))
        # a weighted image
        s = torch.mul(c_, images_t).mean(dim=0)
        s = torch.nn.functional.normalize(s, dim=-1)
        s = s.unsqueeze(3)
        b_ = torch.stack([torch.matmul(images_t[i].unsqueeze(2), s) for i in range(n)])
        b += b_.squeeze()
        c = nn.Softmax(dim=0)(b)
    cs_padded = np.pad(c, pad_width=((0, 0), (padding, padding), (padding, padding)), mode='edge')
    return c, cs_padded


def smooth_weights(c, alpha=0.5):
    smooth_c = np.minimum(c.max(), c + c.max() - np.quantile(c, alpha))
    
    return smooth_c
