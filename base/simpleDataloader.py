import os
import random
import pickle
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

def simpleDataLoader(opt, dataframe_path, transforms=None, shuffle=True):
    """
    A simple dataloader class
        - opt is an argparser
    """
    dataset = simpleDataset(opt=opt, dataframe_path=dataframe_path, transform=transforms)
    if opt.weighted_sampling:
        sampler = simpleWeightedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.workers, sampler = sampler if opt.weighted_sampling else None)

    return dataloader

def simpleWeightedSampler(dataset):
    """
    Makes a weighted sampler based on the input dataset.
    Computes class weights for each class based on its sample distributions.
    Requires the class labels to be numerical and integers.
    """
    # load the dataframe for the dataset and get the weights
    dataset_df = dataset.label
    cls_weights = []
    for i in dataset_df.img_label.unique():
        cls_weights.append(1.0 / dataset_df.img_label[dataset_df.img_label == i].count())

    # make a tensor of the weights for each sample
    sample_weights = []
    for img_cls in dataset_df.img_label:
        sample_weights.append(cls_weights[img_cls])
    sample_weights = torch.tensor(sample_weights)

    return WeightedRandomSampler(sample_weights, len(sample_weights))

class simpleDataset(Dataset):
    """
    A simple custom dataset class
    """
    def __init__(self, opt, dataframe_path, transform=None):
        """
        Args:
            dataframe_path (string): Path to the dataframe with image names, labels, and other information.
                - Can be a pickle (.pkl) or CSV (.csv)
            transform (callable, optional): Optional transform(s) to be applied on a sample.
                - Transforms shoud have the traditional transforms and any on-the-fly processing
        """
        if opt.cls_select is None:
            self.label = pd.read_pickle(dataframe_path) if '.pkl' in dataframe_path else pd.read_csv(dataframe_path)
        else:  # if selecting specific classes
            df = pd.read_pickle(dataframe_path) if '.pkl' in dataframe_path else pd.read_csv(dataframe_path)
            cls_ids = [int(i) for i in opt.cls_select.split(',')]
            df = df[df['img_label'].isin(cls_ids)].reset_index(drop=True)
            for i in range(len(cls_ids)):
                df['img_label'] = df['img_label'].apply(lambda x: i if x == cls_ids[i] else x)
            self.label = df

        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        img_path = os.path.join(self.label['img_path'].iloc[idx])
        image = Image.open(img_path)
        img_label = self.label['img_label'].iloc[idx]
        img_id = self.label['img_id'].iloc[idx]

        if self.transform is not None:
            image = self.transform(image)

        sample = (image, img_label, img_id)

        return sample