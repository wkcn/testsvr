import numpy as np
data = np.loadtxt('data.csv', delimiter = ',', skiprows = 1)
import mxnet as mx
from mxnet import autograd, nd

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

x_i = [4,5,6,7,8,9,10,11]
pred_i = [12,13,14]
x_idx = [1,2,3,4,5,6]
trainX = mx.nd.array(data[x_i][:, x_idx])
trainY = mx.nd.array(data[x_i][:, 7]).reshape((-1, 1))
testX = mx.nd.array(data[pred_i][:, x_idx])
testY = mx.nd.array(data[pred_i][:, 7]).reshape((-1, 1))

def norm(x):
    mean = x.mean(0, keepdims = True)
    std = mx.nd.sqrt((x - mean).square().mean(0, keepdims = True))
    return (x - mean) / std

def norm2(x):
    xmin = x.min(0)
    xmax = x.max(0)
    return (x - xmin) / (xmax - xmin)

trainX = norm(trainX)
testX = norm(testX)
trainY = norm(trainY)
testY = norm(testY)

'''
trainX = mx.nd.ones_like(trainX)
trainY = mx.nd.ones_like(trainY)
testX = mx.nd.ones_like(testX)
testY = mx.nd.ones_like(testY)
'''

from mxnet.gluon import data as gdata
batch_size = 4
dataset = gdata.ArrayDataset(trainX, trainY)
train_iter = gdata.DataLoader(dataset, batch_size, shuffle = True) 
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(32))
net.add(nn.Activation('tanh'))
net.add(nn.Dense(1))

from mxnet import init
net.initialize(init.Normal(sigma = 0.01))

from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()

from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

num_epochs = 10000
for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    n = 0
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        train_loss += l.mean().asnumpy()
        n += 1
        l.backward()
        trainer.step(batch_size)
    print("epoch %d, train_loss: %f, val_loss: %f"
          % (epoch, train_loss / n, loss(net(testX), testY).mean().asnumpy()))
# dense = net[0]
# print (dense.weight.data(), dense.bias.data())
