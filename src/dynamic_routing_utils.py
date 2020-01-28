import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T

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

def smooth_weights(c, alpha=0.5):
    smooth_c = np.minimum(c.max(), c + c.max() - np.quantile(c, alpha))
    
    return smooth_c
