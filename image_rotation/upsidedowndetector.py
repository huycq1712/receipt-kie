import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms

import PIL
from PIL import Image

import os
import glob

class UpsideDowndetector(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model  = mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=2, bias=True)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

    def load(self, path=""):
        self.load_state_dict(torch.load(path))

    def save(self, path=""):
        torch.save(self.state_dict(), path)

if __name__ == "__main__":
    print("ok")