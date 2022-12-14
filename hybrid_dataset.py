import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from PIL.JpegImagePlugin import JpegImageFile

NUM_IMAGES = 31253

class HybridDataset(Dataset):
    def __init__(self, img_dir,
                 albedo_f_prefix="albedo", 
                 low_spp_f_prefix="low_spp_rt", 
                 gt_f_prefix="gt", 
                 transform=None, 
                 target_transform=None):
        self.const_len = NUM_IMAGES
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.albedo_f_prefix = albedo_f_prefix
        self.low_spp_f_prefix = low_spp_f_prefix
        self.gt_f_prefix = gt_f_prefix

    def __len__(self):
        return self.const_len

    def __stack_images(self, low_spp, albedo):
        # print(f"DEBUG: low_spp image has size {low_spp.size()}")
        assert low_spp.size()[0] == 3, f"Expected 3 channels in image, got {low_spp.size()[0]}"
        assert low_spp.size() == albedo.size(), f"Got images with different sizes: {low_spp.size()} and {albedo.size()}"

        six_channel_tensor = torch.zeros((6, low_spp.size()[1], low_spp.size()[2]))
        six_channel_tensor[0:3, :, :] = low_spp
        six_channel_tensor[3:6, :, :] = albedo

        return six_channel_tensor

    def __getitem__(self, idx):

        # setup paths
        albedo_image_path = os.path.join(self.img_dir, f"{self.albedo_f_prefix}_{idx}.jpg")
        low_spp_image_path = os.path.join(self.img_dir, f"{self.low_spp_f_prefix}_{idx}.jpg")
        gt_image_path = os.path.join(self.img_dir, f"{self.gt_f_prefix}_{idx}.jpg")

        # read images
        albedo = read_image(albedo_image_path).float()
        low_spp = read_image(low_spp_image_path).float()
        gt = read_image(gt_image_path).float()
        # albedo = JpegImageFile(albedo_image_path)
        # low_spp = JpegImageFile(low_spp_image_path)
        # gt = JpegImageFile(gt_image_path)

        # apply any transformations before stacking
        if self.transform:
            albedo = self.transform(albedo)
            low_spp = self.transform(low_spp)
            gt = self.transform(gt)

        # stack images (now of H x W x 6):
        # image = self.__stack_images(low_spp, albedo)
        image = low_spp, albedo

        return low_spp, albedo, gt