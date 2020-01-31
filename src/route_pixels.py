# calculate routing weights
import argparse
import os
import glob


from tqdm import tqdm
import numpy as np
import skimage

import torch
import torch.nn as nn
import torchvision.transforms as T

from src.utils import getImageSetDirectories, strip_conf
from src.dynamic_routing_utils import route_per_pixel
from src.predict import get_lr_encodings, load_model
import src.DataLoader as DataLoader 


def save_pixel_routing_scores(dataset_directories, model, its=100):
    '''
    Saves low-resolution routing scores as .npy under imageset dir
    Args:
        dataset_directories: list of imageset directories
    '''

    for imset_dir in tqdm(dataset_directories):
            sample = DataLoader.read_imageset(imset_dir, sorted_k=False)
            sample['lr'] = torch.from_numpy(skimage.img_as_float(sample['lr']).astype(np.float32))
            lrs, alphas = sample['lr'].unsqueeze(0), torch.ones(1, sample['lr'].shape[0])
            weights, hrs, hr_maps, names = None, None, None, None
            encodings = get_lr_encodings((lrs, alphas, weights, hrs, hr_maps, names), model)
            weights = route_per_pixel(torch.from_numpy(encodings), its=its)
            np.save(os.path.join(imset_dir, "weight_maps.npy"), weights)

def main():
    '''
    Calls save_routing on train and test set.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="root dir of the dataset", default='data/')
    parser.add_argument("--model", help="model_path")
    parser.add_argument("--config", help="config_file")
    parser.add_argument("--its", help="number of iterations for the routing", default=100)
    args = parser.parse_args()


    prefix = args.prefix
    its = args.its
    assert os.path.isdir(prefix)
    
    config = strip_conf(args.config)
    model = load_model(config, args.model)
    model.eval()
    if os.path.exists(os.path.join(prefix, "train")):
        train_set_directories = getImageSetDirectories(os.path.join(prefix, "train"))
        save_pixel_routing_scores(train_set_directories, model, its) # train data


    if os.path.exists(os.path.join(prefix, "test")):
        test_set_directories = getImageSetDirectories(os.path.join(prefix, "test"))
        save_pixel_routing_scores(test_set_directories, model, its) # test data



if __name__ == '__main__':
    main()
