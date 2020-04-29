# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:26:55 2020

@author: MYM
"""
import matplotlib.pyplot as plt
from mnist import load_mnist
import numpy as np 
from collections import OrderedDict
def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
def softmax(x):
    if x.ndim==1:
        x=x-np.max(x)
        out= np.exp(x)/np.sum(np.exp(x))
    else:
        x=x.T
        x=x-np.max(x,axis=0)
        out=np.exp(x)/np.sum(np.exp(x),axis=0)
        out=out.T
    return out
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size    

class Affine:
    def __init__(self,w,b):
        self.w=w
        self.b=b
        self.x=None
        self.out=None
        self.dx=None
        self.dw=None
        self.db=None
    def forward(self,x):
        self.x=x
        self.out=np.dot(x,self.w)+self.b
        return self.out
    def backward(self,dout):
        self.dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return self.dx
class Relu:
    def __init__(self):
        self.out=None
        self.index=None
        self.dx=None
    def forward(self,x):
        self.index=(x<0)
        self.out=relu(x)
        return self.out
    def backward(self,dout):
        dout[self.index]=0#这里回头再看 矩阵维度问题
        self.dx=dout
        return self.dx
class softmaxwithloss:
    def __init__(self):
        self.out=None
        self.loss=None
        self.dx=None
        self.t=None
    def forward(self,x,t):
        self.t=t
        self.out=softmax(x)
        self.loss=cross_entropy_error(self.out,t)
        return self.loss
    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        self.dx=dout*(self.out-self.t)
        return self.dx/batch_size
class network:
    def __init__(self,input_size,hid_size,out_size,std):
        w1=std*np.random.randn(input_size,hid_size)
        b1=np.zeros(hid_size)
        w2=std*np.random.randn(hid_size,out_size)
        b2=np.zeros(out_size)
        self.para={}
        self.para['w1']=w1
        self.para['b1']=b1
        self.para['w2']=w2
        self.para['b2']=b2
        self.layers = OrderedDict()
        self.layers['Affine1']=Affine(w1,b1)
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(w2,b2)
        self.lastlayer=softmaxwithloss()
    def predict(self,x):
        for key in self.layers.values():
            x=key.forward(x)
        return x
    def loss(self,x,t):
        y=self.predict(x)
        loss=self.lastlayer.forward(y,t)
        return loss
    def backward(self,x,t):
        loss=self.loss(x,t)
        dout=self.lastlayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for key in layers:
            dout=key.backward(dout)
        grad={}
        grad['dw1']=self.layers['Affine1'].dw
        grad['dw2']=self.layers['Affine2'].dw
        grad['db1']=self.layers['Affine1'].db
        grad['db2']=self.layers['Affine2'].db
        return grad
    def accuracy(self, x, t):
        y=self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
        
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
mym_network=network(784,60,10,0.01)
iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad=mym_network.backward(x_batch,t_batch)
    mym_network.layers['Affine1'].w= mym_network.layers['Affine1'].w-learning_rate*grad['dw1']
    mym_network.layers['Affine2'].w= mym_network.layers['Affine2'].w-learning_rate*grad['dw2']
    mym_network.layers['Affine1'].b=mym_network.layers['Affine1'].b-learning_rate*grad['db1']
    mym_network.layers['Affine2'].b=mym_network.layers['Affine2'].b-learning_rate*grad['db2']
    
    loss = mym_network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    
    if i % iter_per_epoch == 0:
        train_acc = mym_network.accuracy(x_train, t_train)
        test_acc = mym_network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()