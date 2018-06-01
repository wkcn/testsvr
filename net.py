import numpy as np
data = np.loadtxt('data.csv', delimiter = ',', skiprows = 1)
import mxnet as mx

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

x_i = [4,5,6,7,8,9,10,11]
pred_i = [12,13,14]
x_idx = [7]#[1,2,3,4,5,6,7]
trainX = mx.nd.array(data[x_i][:, x_idx])
trainY = mx.nd.array(data[x_i][:, 7]).reshape((-1, 1))
testX = mx.nd.array(data[pred_i][:, x_idx])
testY = mx.nd.array(data[pred_i][:, 7]).reshape((-1, 1))

def norm(x):
    xmin = x.min(0)
    xmax = x.max(0)
    return (x - xmin) / (xmax - xmin)

trainX = norm(trainX)
testX = norm(testX)

batch_size = 3
train_iter = mx.io.NDArrayIter(trainX, trainY, batch_size = batch_size)
val_iter = mx.io.NDArrayIter(testX, testY, batch_size = batch_size)

x = mx.sym.var('data')
y = mx.sym.var('softmax_label')

'''
x = mx.sym.FullyConnected(data = x, num_hidden = 30)
x = mx.sym.relu(data = x)
'''

x = mx.sym.FullyConnected(data = x, num_hidden = 1)

loss = mx.sym.mean(mx.sym.square(x - y))
loss = mx.sym.MakeLoss(loss)

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on compute context
mlp_model = mx.mod.Module(symbol=loss, context=ctx)
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':1e-5},  # use fixed learning rate
              eval_metric='mae',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 1), # output progress for each 100 data batches
              num_epoch=1000)  # train for at most 10 dataset passes
