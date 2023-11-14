import torch
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import label_binarize
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class Net(torch.nn.Module):
    def __init__(self, dimension, input_size, num_layers, output_size=10):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=dimension, num_layers=num_layers)
        self.fc = torch.nn.Linear(dimension, output_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        c, _ = self.lstm(x)
        c = self.fc(c[:,-1])
        c = self.softmax(c)
        return c

data_train = arff.loadarff('/home/shangar21/Downloads/InsectSound/InsectSound_TRAIN.arff')
data_test = arff.loadarff('/home/shangar21/Downloads/InsectSound/InsectSound_TEST.arff')

df = pd.DataFrame(data_train[0])
df_data = df.loc[:, df.columns != 'target']
df_labels = df.loc[:, df.columns == 'target']
unique_labels = list(np.unique(df_labels['target']))
df_labels['target'] = df_labels['target'].apply(lambda x: unique_labels.index(x))

labels = label_binarize(df_labels.to_numpy(), classes=[i for i in range(len(unique_labels))])

X_train, _, y_train, _ = train_test_split(df_data.to_numpy(), labels, train_size=0.999)

net = Net(dimension=600, input_size=1, num_layers=1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

EPOCHS = 2

for _ in range(EPOCHS):
    for i in tqdm(range(len(X_train))):
        x = torch.tensor(X_train[i]).reshape(-1, 1).type(torch.float32)
        optimizer.zero_grad()
        output = net(x)
        t = torch.tensor(y_train[i]).type(torch.float32)
        loss = criterion(output, t)
        loss.backward()
        optimizer.step()

correct = 0
total = 0

for i in tqdm(range(len(X_train))):
    x = torch.tensor(X_train[i]).reshape(-1, 1).type(torch.float32)
    output = torch.argmax(net(x))
    correct += 1 if labels[i][output] == 1 else 0
    total += 1

print("Accuracy: ", correct/total)
