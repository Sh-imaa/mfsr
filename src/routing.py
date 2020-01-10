# calculate routing weights
import argparse
import os
import glob

from tqdm import tqdm
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T

from utils import getImageSetDirectories

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


def save_routing_scores(dataset_directories, its=500):
    '''
    Saves low-resolution routing scores as .npy under imageset dir
    Args:
        dataset_directories: list of imageset directories
    '''

    for imset_dir in tqdm(dataset_directories):

        idx_names = np.array([os.path.basename(path)[2:-4] for path in glob.glob(os.path.join(imset_dir, 'LR*.png'))])
        idx_names = np.sort(idx_names)
        tns = [T.functional.to_tensor(cv2.imread(os.path.join(imset_dir, f'LR{i}.png'))) for i in idx_names]
        weights = route(tns, its)
        np.save(os.path.join(imset_dir, "routing_weights.npy"), weights)

def main():
    '''
    Calls save_routing on train and test set.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="root dir of the dataset", default='data/')
    parser.add_argument("--its", help="number of iterations for the routing", default=500)
    args = parser.parse_args()


    prefix = args.prefix
    its = args.its
    assert os.path.isdir(prefix)
    if os.path.exists(os.path.join(prefix, "train")):
        train_set_directories = getImageSetDirectories(os.path.join(prefix, "train"))
        save_routing_scores(train_set_directories, its) # train data


    if os.path.exists(os.path.join(prefix, "test")):
        test_set_directories = getImageSetDirectories(os.path.join(prefix, "test"))
        save_routing_scores(test_set_directories, its) # test data



if __name__ == '__main__':
    main()
