import torch.nn as nn
import torch.nn.functional as F
from shape_generator import generate_objects
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torch

def bbox_to_min_max(bbox, shape):
    rect = bbox[np.where(shape == 0)][0]
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

def shape_to_id_vec(shape):
    vec = np.zeros(3)
    for i in shape:
        vec[i] = 1
    return vec

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 1, 5)
        self.fc1 = nn.Linear(25, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

bboxes, imgs, shapes, colors = generate_objects(8000, 32, 4, 16, 2)
#X_train, X_test, Y_train, Y_test = train_test_split(imgs, bboxes, test_size=0.33, random_state=42)

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
transform = transforms.ToTensor()

#print(bboxes[0])
#print(imgs[0].shape)
#import matplotlib.pyplot as plt
#plt.imshow(imgs[0], interpolation='none', origin='lower', extent=[0, 32, 0, 32])
#plt.show()

for epoch in range(2):
    running_loss = 0.0
    for x in range(len(imgs)):
        inputs = transform(Image.fromarray(imgs[x]))
        labels = torch.reshape(torch.tensor(shape_to_id_vec(shapes[x])), (1, 3)).type(torch.FloatTensor)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if x % 2000 == 1999:
            print('loss: ', running_loss/2000)
            running_loss = 0
