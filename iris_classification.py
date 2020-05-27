#following sentdex tutorial
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(4, 5)
		self.fc2 = nn.Linear(5, 5)
		self.fc3 = nn.Linear(5,3)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)



data = load_iris(True)
in_data, out_data = shuffle(data[0], data[1])
train = []

for i in range(len(in_data)):
	insert = [in_data[i], out_data[i]]
	train.append(insert)

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)


# for data in trainset:
# 	break

# X, y = data
# print(X.shape)
# print(X.view(-1,4).shape)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.1)

EPOCHS = 3

for epoch in range(EPOCHS):
	for data in trainset:
		X, y = data
		net.zero_grad()
		output = net(X.float())
		loss = F.nll_loss(output, y)
		loss.backward()
		optimizer.step()

correct = 0
total = 0

with torch.no_grad():
	 for data in trainset:
	 	X, y = data
	 	output = net(X.float())
	 	for idx, i in enumerate(output):
	 		if torch.argmax(i) == y[idx]:
	 			correct += 1
	 		total += 1

print("Accuracy: ", correct/total)
