import torch.nn as nn
import torch.nn.functional as F
from shape_generator import generate_objects
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torch

class BBoxRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32, 4)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

