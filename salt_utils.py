import torch
import torchvision.transforms.functional as F_img
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from random import random

class SaltLoader(Dataset):
  def __init__(self, img_folder, img_mask_folder, depth_file):
    img_ids = [file.split('/')[3][:-4] for file in glob(img_folder+'*.png')]
    img_paths = []
    img_mask_paths = []
    
    depth_df = pd.read_csv(depth_file, index_col=0)
    depth_arr = np.empty(len(img_ids))    
    
    for i, img_id in enumerate(img_ids):
      img_paths.append(img_folder+img_id+'.png')
      img_mask_paths.append(img_mask_folder+img_id+'.png')
      depth_arr[i] = depth_df.loc[img_id]
    
    self.img_paths = img_paths
    self.img_mask_paths = img_mask_paths
    self.depth_arr = depth_arr
    self.depth_mean, self.depth_std = 506.45332, 208.60599
    self.img_mean, self.img_std = 122.80216, 37.67669
    
  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, idx):
    img = F_img.to_grayscale(Image.open(self.img_paths[idx])) 
    img_mask = F_img.to_grayscale(Image.open(self.img_mask_paths[idx]))
    
    if random() > .5:
      img = F_img.hflip(img)
      img_mask = F_img.hflip(img_mask)
    if random() > .5:
      img = F_img.vflip(img)
      img_mask = F_img.vflip(img_mask)
    
    img = F_img.resize(img, 128)
    img_mask = F_img.resize(img_mask, 128)
    
    depth = torch.full([1, 128, 128], (self.depth_arr[idx] - self.depth_mean) / self.depth_std)
    return torch.cat([(F_img.to_tensor(img) - self.img_mean) / self.img_std, depth]), F_img.to_tensor(img_mask)