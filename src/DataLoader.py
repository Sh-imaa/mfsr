""" Python script to load, augment and preprocess batches of data """

from collections import OrderedDict
import numpy as np
from os.path import join, exists, basename, isfile

import glob
import skimage
from skimage import io

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dynamic_routing_utils import smooth_weights, get_routing_clusters


def get_patch(img, x, y, size=32):
    """
    Slices out a square patch from `img` starting from the (x,y) top-left corner.
    If `im` is a 3D array of shape (l, n, m), then the same (x,y) is broadcasted across the first dimension,
    and the output has shape (l, size, size).
    Args:
        img: numpy.ndarray (n, m), input image
        x, y: int, top-left corner of the patch
        size: int, patch size
    Returns:
        patch: numpy.ndarray (size, size)
    """
    
    patch = img[..., x:(x + size), y:(y + size)]   # using ellipsis to slice arbitrary ndarrays
    return patch


class ImageSet(OrderedDict):
    """
    An OrderedDict derived class to group the assets of an imageset, with a pretty-print functionality.
    """

    def __init__(self, *args, **kwargs):
        super(ImageSet, self).__init__(*args, **kwargs)

    def __repr__(self):
        dict_info = f"{'name':>10} : {self['name']}"
        for name, v in self.items():
            if hasattr(v, 'shape'):
                dict_info += f"\n{name:>10} : {v.shape} {v.__class__.__name__} ({v.dtype})"
            else:
                dict_info += f"\n{name:>10} : {v.__class__.__name__} ({v})"
        return dict_info


def sample_clearest(weights, n=None, beta=50, seed=None):
    """
    Given a set of clearances, samples `n` indices with probability proportional to their clearance.
    Args:
        weights: numpy.ndarray, goodnes scores
        n: int, number of low-res views to read
        beta: float, inverse temperature. beta 0 = uniform sampling. beta +infinity = argmax.
        seed: int, random seed
    Returns:
        i_sample: numpy.ndarray (n), sampled indices
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    e_c = np.exp(beta * weights / weights.max()) ##### FIXME: This is numerically unstable. 
    p = []
    e_c_sum = e_c.sum()
    p = e_c / e_c_sum
    nans_num = np.isnan(p).sum()
    if nans_num > 0:
        p[np.isnan(p)] = 1 / nans_num
    idx = range(len(p))
    i_sample = np.random.choice(idx, size=n, p=p, replace=False)
    return i_sample

def read_imageset(imset_dir, create_patches=False, patch_size=64, seed=None,
                  top_k=None, beta=0., lr_weights="random", outlier="keep",
                  sorted_k=False):
    """
    Retrieves all assets from the given directory.
    Args:
        imset_dir: str, imageset directory.
        create_patches: bool, samples a random patch or returns full image (default).
        patch_size: int, size of low-res patch.
        top_k: int, number of low-res views to read.
            If top_k = None (default), low-views are loaded in the order of goodness scores.
            Otherwise, top_k views are sampled with probability proportional to their goodnes.
        beta: float, parameter for random sampling of a reference proportional to its goodnes.
        load_lr_maps: bool, reads the status maps for the LR views (default=True).
        lr_weights: str, how to sample lrs
    Returns:
        dict, collection of the following assets:
          - name: str, imageset name.
          - lr: numpy.ndarray, low-res images.
          - hr: high-res image.
          - hr_map: high-res status map.
          - weight: precalculated average goodness scores
    """

    # Read asset names
    idx_names = np.array([basename(path)[2:-4] for path in glob.glob(join(imset_dir, 'QM*.png'))])
    idx_names = np.sort(idx_names)
    
    # default is random, where every LR gets equal weight (np.ones)
    weights = np.ones(len(idx_names))
    if lr_weights == "clearance":
        if isfile(join(imset_dir, 'clearance.npy')):
            try:
                weights = np.load(join(imset_dir, 'clearance.npy'))  # load clearance scores
            except Exception as e:
                print("please call save_clearance.py before calling DataLoader")
                print(e)
        else:
            raise Exception("please call save_clearance.py before calling DataLoader")

    elif lr_weights == "routing":
        if isfile(join(imset_dir, 'routing_weights.npy')):
            try:
                weights = np.load(join(imset_dir, 'routing_weights.npy'))  # load routing scores
                weights = weights.squeeze()
            except Exception as e:
                print("please call routing.py before calling DataLoader")
                print(e)
        else:
            raise Exception("please call routing.py before calling DataLoader")

    elif lr_weights == "smooth_routing":
        if isfile(join(imset_dir, 'routing_weights.npy')):
            try:
                weights = np.load(join(imset_dir, 'routing_weights.npy'))  # load routing scores
                weights = weights.squeeze()
                weights = smooth_weights(weights)
            except Exception as e:
                print("please call routing.py before calling DataLoader")
                print(e)
        else:
            raise Exception("please call routing.py before calling DataLoader")

    if sorted_k and (top_k is not None) and (top_k > 0):
        top_k = min(top_k, len(idx_names))
        i_clear_sorted = np.argsort(weights)[::-1][:top_k]  # max to min
        weights = weights[i_clear_sorted]
        idx_names = idx_names[i_clear_sorted]
    
    elif top_k is not None and top_k > 0:
        top_k = min(top_k, len(idx_names))
        i_samples = sample_clearest(weights, n=top_k, beta=beta, seed=seed)
        idx_names = idx_names[i_samples]
        weights = weights[i_samples]

    else:
        i_clear_sorted = np.argsort(weights)[::-1]  # max to min
        weights = weights[i_clear_sorted]
        idx_names = idx_names[i_clear_sorted]

    if outlier == "remove":
        if isfile(join(imset_dir, 'routing_weights.npy')):
            try:
                weights_ = np.load(join(imset_dir, 'routing_weights.npy'))  # load routing scores
                weights_ = weights_.squeeze()
                _, good_indeces, _ = get_routing_clusters(weights_)
                weights = weights[:len(good_indeces)]
                good_indeces = ['{0:03}'.format(i) for i in good_indeces]
                idx_names = [i for i in idx_names if i in good_indeces]

            except Exception as e:
                print("please call routing.py before calling DataLoader")
                print(e)

    if outlier == "replace":
        if isfile(join(imset_dir, 'routing_weights.npy')):
            try:
                weights_ = np.load(join(imset_dir, 'routing_weights.npy'))  # load routing scores
                weights_ = weights_.squeeze()
                bad_indeces, good_indeces, _ = get_routing_clusters(weights_)
                weights = np.concatenate((weights[:len(good_indeces)], weights[:len(bad_indeces)])) 
                good_indeces = ['{0:03}'.format(i) for i in good_indeces]
                idx_names = [i for i in idx_names if i in good_indeces]
                idx_names += idx_names[: len(bad_indeces)]

            except Exception as e:
                print("please call routing.py before calling DataLoader")
                print(e)


    lr_images = np.array([io.imread(join(imset_dir, f'LR{i}.png')) for i in idx_names], dtype=np.uint16)
    hr_map = np.array(io.imread(join(imset_dir, 'SM.png')), dtype=np.bool)
    if exists(join(imset_dir, 'HR.png')):
        hr = np.array(io.imread(join(imset_dir, 'HR.png')), dtype=np.uint16)
    else:
        hr = None  # no high-res image in test data

    if create_patches:
        if seed is not None:
            np.random.seed(seed)

        max_x = lr_images[0].shape[0] - patch_size
        max_y = lr_images[0].shape[1] - patch_size
        x = np.random.randint(low=0, high=max_x)
        y = np.random.randint(low=0, high=max_y)
        lr_images = get_patch(lr_images, x, y, patch_size)  # broadcasting slicing coordinates across all images
        hr_map = get_patch(hr_map, x * 3, y * 3, patch_size * 3)

        if hr is not None:
            hr = get_patch(hr, x * 3, y * 3, patch_size * 3)


    # Organise all assets into an ImageSet (OrderedDict)
    imageset = ImageSet(name=basename(imset_dir),
                        lr=np.array(lr_images),
                        hr=hr,
                        hr_map=hr_map,
                        weights=(weights / weights.max()),
                        )

    return imageset




class ImagesetDataset(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories."""

    def __init__(self, imset_dir, config, seed=None, top_k=-1, beta=0.):

        super().__init__()
        self.imset_dir = imset_dir
        self.name_to_dir = {basename(im_dir): im_dir for im_dir in imset_dir}
        self.create_patches = config["create_patches"]
        self.patch_size = config["patch_size"]
        self.seed = seed  # seed for random patches
        self.top_k = top_k
        self.beta = beta
        self.outlier = config["outlier"]
        self.lr_weights = config["lr_weights"]
        self.sorted_k = config["sorted_k"]
        
    def __len__(self):
        return len(self.imset_dir)        

    def __getitem__(self, index):
        """ Returns an ImageSet dict of all assets in the directory of the given index."""    

        if isinstance(index, int):
            imset_dir = [self.imset_dir[index]]
        elif isinstance(index, str):
            imset_dir = [self.name_to_dir[index]]
        elif isinstance(index, slice):
            imset_dir = self.imset_dir[index]
        else:
            raise KeyError('index must be int, string, or slice')

        imset = [read_imageset(imset_dir=dir_,
                               create_patches=self.create_patches,
                               patch_size=self.patch_size,
                               seed=self.seed,
                               top_k=self.top_k,
                               beta=self.beta,
                               outlier=self.outlier,
                               lr_weights=self.lr_weights,
                               sorted_k=self.sorted_k)
                    for dir_ in tqdm(imset_dir, disable=(len(imset_dir) < 11))]

        if len(imset) == 1:
            imset = imset[0]

        imset_list = imset if isinstance(imset, list) else [imset]
        for i, imset_ in enumerate(imset_list):
            imset_['lr'] = torch.from_numpy(skimage.img_as_float(imset_['lr']).astype(np.float32))
            if imset_['hr'] is not None:
                imset_['hr'] = torch.from_numpy(skimage.img_as_float(imset_['hr']).astype(np.float32))
                imset_['hr_map'] = torch.from_numpy(imset_['hr_map'].astype(np.float32))
            imset_list[i] = imset_

        if len(imset_list) == 1:
            imset = imset_list[0]

        return imset
