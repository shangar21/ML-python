import pandas as pd
import matplotlib
from sklearn import svm 
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("wine.csv")
df = np.array(df)

y = np.concatenate(df[:, [0]])
y = np.array([0 if i != 1 else 1 for i in y])
x = np.delete(df, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y)

rbf_svc = svm.SVC(kernel='rbf')

rbf_svc.fit(X_train, y_train)

count = 0

for i in range(len(X_test)):
	print(rbf_svc.predict(X_test[i].reshape(1, -1))[0], y_test[i])
	if rbf_svc.predict(X_test[i].reshape(1, -1))[0] == y_test[i]:
		count += 1


print('accuracy: {}'.format(count/len(X_test)))

