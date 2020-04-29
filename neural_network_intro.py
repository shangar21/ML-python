# Following sentdex tutorial for deep learning using pytorch 

import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# fetching data from built in data set from pytorch MNIST (handwritten digits)
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])) 
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])) 

#loading data from fetched data using pytorch's data loader
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# for data in trainset:
# 	#print(data)
# 	break

print(trainset)

# plt.imshow(data[0][0].view(28,28)) #apparantly shape of data = (1,28,28).... weird right? so you get errors when trying to output... wonder why that 1 is there by boi left me in suspense
# plt.show()



