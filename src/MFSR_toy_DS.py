import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def random_homography(img, scale=0.02):
    np_img = np.array(img)
    np_img = np.moveaxis(np_img, 0, -1)
    w, h = np_img.shape[1:3]
    points = np.array([(0, 0), (0, h), (w, 0), (w, h)])
    points_trans = points + np.random.normal(size=points.shape, scale=scale)

    h_matrix, _ = cv2.findHomography(points, points_trans)
    new_img = cv2.warpPerspective(np_img, h_matrix, dsize=np_img.shape[:2])

    return torch.from_numpy(new_img)


def trans_and_scale(img, rot=10, trans=(0.05, 0.05),
                    homography=False, random_interpolation=False):
    h, w = np.array(img).shape[1:3]
    transformation = []
    trans_interpolation = 2
    if homography:
        transformation += [T.Lambda(random_homography),
                           T.ToPILImage()]
    else: 
        transformation += [T.ToPILImage(),
                           T.RandomAffine(rot, trans, resample=trans_interpolation)]
    # choose Bicubic if not random interpolation
    interpolation = random.choice(range(1, 6)) if random_interpolation else 4
    transformation.append(T.Resize(h//2, interpolation=interpolation))
    transformation.append(T.ToTensor())
    transformation = T.Compose(transformation)

    return transformation(img)

class MFSRDataSet(Dataset):
    def __init__(self, data, views=2, max_view=False,
                 blur=True, to_grey=False, homography=False):
        super().__init__()
        # same set of images whenever an instantiation happens
        numpy.random.seed(29)
        random.seed(29)
        self.data = data
        self.views = views
        self.homography = homography
        self.to_grey = to_grey
        self.blur = blur
        self.max_view = max_view
    
    def __getitem__(self, i):
        lrs = []
        orig_img = self.data[i][0]

        if self.to_grey:
          # get first channel instead of grey
          orig_img = orig_img[0].unsqueeze(0)
        
        if self.blur:
          img = np.array(orig_img)
          img = cv2.blur(img, (3, 3))
          img = torch.from_numpy(img)
        else:
          # get a copy not a refrence
          img = np.array(orig_img)

        n_views = np.random.randint(low=1, high=self.views + 1) if not self.max_view else self.views

        for _ in range(n_views):
            lr = trans_and_scale(img, rot=0, homography=self.homography)
            lrs.append(lr)

        # pad lrs in case of elements less than max_view
        lrs += [torch.zeros_like(lrs[0]) for _ in range(self.views - n_views)]
        alphas = torch.zeros((self.views))
        alphas[:n_views] = 1
        return orig_img, torch.stack(lrs).squeeze(1), alphas
    
    def __len__(self):
        return len(self.data)