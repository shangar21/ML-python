# Following sentdex tutorial for deep learning using pytorch 

import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn #oop
import torch.nn.functional as F 
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return F.log_softmax(x, dim=1)

# fetching data from built in data set from pytorch MNIST (handwritten digits)
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])) 
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])) 
<<<<<<< HEAD
=======

>>>>>>> 00c51a5442762eaf1847979bc93c1de93be3f97a
#loading data from fetched data using pytorch's data loader
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

<<<<<<< HEAD
# for data in trainset:
# 	break
# X, y = data
# print(X.shape)
# print(X.view(-1, 28*28).shape)

=======
>>>>>>> 00c51a5442762eaf1847979bc93c1de93be3f97a
net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
	for data in trainset:
		X, y = data
		net.zero_grad()
		output = net(X.view(-1, 28*28))
<<<<<<< HEAD
		print(output)
=======
>>>>>>> 00c51a5442762eaf1847979bc93c1de93be3f97a
		loss = F.nll_loss(output, y)
		loss.backward()
		optimizer.step()

correct = 0
total = 0

with torch.no_grad():
	 for data in trainset:
	 	X, y = data
	 	output = net(X.view(-1,28*28))
	 	for idx, i in enumerate(output):
	 		if torch.argmax(i) == y[idx]:
	 			correct += 1
	 		total += 1

print("Accuracy: ", correct/total)

