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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Training on {}".format(device))

data_train = arff.loadarff('/home/shangar21/Downloads/InsectSound/InsectSound_TRAIN.arff')
data_test = arff.loadarff('/home/shangar21/Downloads/InsectSound/InsectSound_TEST.arff')

df = pd.DataFrame(data_train[0])
df_data = df.loc[:, df.columns != 'target']
df_labels = df.loc[:, df.columns == 'target']
unique_labels = list(np.unique(df_labels['target']))
df_labels['target'] = df_labels['target'].apply(lambda x: unique_labels.index(x))

labels = label_binarize(df_labels.to_numpy(), classes=[i for i in range(len(unique_labels))])

X_train, _, y_train, _ = train_test_split(df_data.to_numpy(), labels, train_size=0.999)

clip_len = 40
net = Net(dimension=clip_len, input_size=1, num_layers=1)
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
criterion = torch.nn.BCELoss()

EPOCHS = 1000

running_loss = 0

#X_train = X_train[:2]

for _ in range(EPOCHS):
    for i in tqdm(range(len(X_train))):
        x = torch.tensor(X_train[i]).reshape(-1, 1).type(torch.float32)[:clip_len].to(device)
        optimizer.zero_grad()
        output = net(x)
        t = torch.tensor(y_train[i]).type(torch.float32).to(device)
        loss = criterion(output, t)
        running_loss += loss
        loss.backward()
        optimizer.step()
    if _ % 20 == 19:
        print(f"running loss: {running_loss}")
        running_loss = 0

correct = 0
total = 0

for i in tqdm(range(len(X_train))):
    x = torch.tensor(X_train[i]).reshape(-1, 1).type(torch.float32)[:clip_len].to(device)
    output = torch.argmax(net(x))
    print(net(x))
    correct += 1 if labels[i][output] == 1 else 0
    total += 1

print("Accuracy: ", correct/total)
