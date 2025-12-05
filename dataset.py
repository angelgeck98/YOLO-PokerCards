'''
Card classifier using Kaggle dataset of 224x224x3 images

Perhaps we don't need this file if we use the 
dataset.yaml thing for YOLO 
Link: https://docs.ultralytics.com/datasets/detect/
dataset configuration we'll do ^^
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import kagglehub
from kagglehub import KaggleDatasetAdapter

# --- 1. Dataset & Augmentation ---

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label
    
    @property
    def classes(self):
        return self.data.classes
    
'''
Not quite sure what to put here yet, or if we should put it elsewhere.
Include all dataset images as zip files?? or just get from the site?
'''
def load_data():
    dataset = PlayingCardDataset(
        data_dir='/kaggle/input/cards-image-datasetclassification/train'
    )