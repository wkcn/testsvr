import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

data = np.loadtxt('5.csv', delimiter = ',', skiprows = 1)

ratio = 0.7
len_train_set = int(round(len(data) * ratio))

train_set = data[len(data)-len_train_set:]
val_set = data[:len(data)-len_train_set]

train_X = train_set[:, 2:]
train_y = train_set[:, 1]

val_X = val_set[:, 2:]
val_y = val_set[:, 1]

clf = SVR(C = 1.0, epsilon = 0.1, kernel = 'poly', verbose = False)
# clf = LinearRegression()
clf.fit(train_X, train_y)

predict_y = clf.predict(val_X)
print (predict_y, val_y)

print (np.sqrt(np.mean((np.square(predict_y - val_y)))))
