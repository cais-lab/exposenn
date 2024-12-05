import os

from typing import Union, Iterable

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import pandas as pd


class AnnotatedImagesDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading images and their associated labels from a CSV or DataFrame of annotations.
    Each sample consists of an image, multiple labels (e.g., concept annotations), and a target label.

    This dataset class is designed to work with structured annotations for images, where each image is linked to
    several labels (e.g., for multi-task learning or concept-based models). Images are loaded from disk, and corresponding
    labels are loaded from a DataFrame or CSV file.

    Parameters
    ----------
    annotations : Union[str, pd.DataFrame]
        The annotations for the dataset, either as a path to a CSV file or a pandas DataFrame. The annotations should
        contain columns for the image filenames, labels, and target values.
    img_dir : str
        The directory where the images are stored.
    name_column : str
        The column in the annotations that contains the image filenames.
    target_column : str
        The column in the annotations that contains the target label for each image.
    label_columns : list[str]
        A list of columns in the annotations that contain additional labels for each image. These labels could be
        concept labels or any other auxiliary annotations.
    transform : callable, optional
        An optional transformation function (e.g., from `torchvision.transforms`) that will be applied to each image
        after loading. This can be used for data augmentation or preprocessing.

    Attributes
    ----------
    img_labels : pd.DataFrame
        A DataFrame containing the image filenames, labels, and target values, filtered to only include the relevant
        columns (`name_column`, `label_columns`, and `target_column`).
    img_dir : str
        The directory where the images are stored.
    name_column : str
        The column name for image filenames in the annotations.
    target_column : str
        The column name for the target labels in the annotations.
    label_columns : list[str]
        The list of column names for the additional labels in the annotations.
    transform : callable or None
        The transformation function to be applied to each image, or `None` if no transformations are provided.

    Methods
    -------
    __len__()
        Returns the total number of samples in the dataset (i.e., the number of rows in the annotations).
    __getitem__(idx)
        Returns a tuple (image, labels, target) for the sample at the given index `idx`. The image is loaded from disk,
        the labels are the additional labels as a tensor, and the target is the main label for the image.
    """

    def __init__(self, annotations: Union[str, pd.DataFrame], 
                       img_dir: str,
                       name_column: str,
                       target_column: str,
                       label_columns: list[str],
                       transform=None):
        
        if type(annotations) == str:
            self.img_labels = pd.read_csv(annotations)
        else:
            self.img_labels = annotations.copy(deep=True)
            
        assert name_column in self.img_labels.columns
        assert target_column in self.img_labels.columns
        for col in label_columns:
            assert col in self.img_labels.columns
        
        self.name_column = name_column
        self.target_column = target_column
        self.label_columns = label_columns
        selected_columns = [self.name_column] + self.label_columns + [self.target_column]
        
        self.img_labels = self.img_labels[selected_columns]
        self.img_dir = img_dir
        self.transform = transform        

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """
        Returns a tuple (image, labels, target) for the sample at index `idx`.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        image : PIL.Image.Image
            The image loaded from disk, potentially transformed by `self.transform`.
        labels : torch.Tensor
            The additional labels associated with the image as a tensor.
        target : torch.Tensor
            The main target label for the image as a tensor.

        Raises
        ------
        FileNotFoundError
            If the image file cannot be found at the specified path.
        """

        img_path = os.path.join(self.img_dir, self.img_labels[self.name_column].iloc[idx])
        image = pil_loader(img_path)
        labels = torch.from_numpy(self.img_labels[self.label_columns].iloc[idx, :].to_numpy(dtype=np.int8))
        tgt = torch.tensor([self.img_labels[self.target_column].iloc[idx]])
        if self.transform:
            image = self.transform(image)
        return image, labels, tgt
