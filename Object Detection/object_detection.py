import torch.nn as nn
import torch.nn.functional as F
from shape_generator import generate_objects
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torch
from classifier import Net
from bbox_regressor import BBoxRegressor

def bbox_to_min_max(bbox, shape):
    rect = bbox[np.where(shape == 0)][0]
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

def shape_to_id_vec(shape):
    vec = np.zeros(3)
    for i in shape:
        vec[i] = 1
    return vec

def train(net_class, train_input, train_output, format_func=lambda x: x):
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    transform = transforms.ToTensor()

    for epoch in range(2):
        running_loss = 0.0
        for x in range(len(train_input)):
            inputs = transform(Image.fromarray(imgs[x]))
            labels = format_func(torch.tensor(train_output[x])).type(torch.FloatTensor)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if x % 2000 == 1999:
                print('loss: ', running_loss/2000)
                running_loss = 0
    return net

if __name__ == '__main__':
    bboxes, imgs, shapes, colors = generate_objects(8000, 32, 4, 16, 2)
    shapes = [shape_to_id_vec(i) for i in shapes]
#    net = train(Net, imgs, shapes, format_func=lambda x : torch.reshape(x, (1,3)))
    bbox_regressor = train(BBoxRegressor, imgs, bboxes)


