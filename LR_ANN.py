#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir= 'C:\\Users\\Administrator\\Desktop\\Task3\\t3.csv'
x=pd.read_csv(dir)
x.head()





x.corr()





x.isna().sum()
x.ffill(inplace=True)
x.isna().sum()




x1 = x['Duration'].values
y1 = x['Calories'].values
N=len(x1)




#Splitting Data into train and validation
idx=np.arange(N)
np.random.shuffle(idx)
idx_train=idx[:int(0.8*N)]
idx_test=idx[int(0.8*N):]
x_train, y_train = x1[idx_train],y1[idx_train]
x_val, y_val = x1[idx_test],y1[idx_test]


#plotting tain and val data
plt.figure('1')
plt.scatter(x_train,y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.figure('2')
plt.scatter(x_val,y_val,color = 'm')
plt.xlabel('x_val')
plt.ylabel('y_val')
plt.show(block=True)





#training loop
#initializing parameters
trainLosses=[]
valLosses=[]
lr=0.0001
#w=np.random.randn(1)
#b=np.random.randn(1)
w=4
b=0.1
print(w)
print(b)





for i in range(20):
    #forward pass
    yhat=w*x_train+b #note vectorized operation
    #MSE loss
    error=yhat-y_train
    loss= (error**2).mean()
    trainLosses.append(loss)
    #computing gradients
    db=2*error.mean()
    dw=2*(x_train*error).mean()
    #weight update
    b=b-lr*db
    w=w-lr*dw

    #val MSE loss
    yhatVal=w*x_val+b
    errorVal=yhatVal-y_val
    valLoss= (errorVal**2).mean()
    valLosses.append(valLoss)

    #stopping condition
    if(valLoss<0.0001):
        break

    print(f'train loss={loss}, val loss={valLoss}, w={w}, b={b}')

    #training data plot
    plt.figure('3')
    plt.cla()
    plt.scatter(x_train,y_train)
    plt.scatter(x_train,yhat)
    plt.title(f'epoch={i}, loss={loss}, w={w}, b={b}')
    plt.show(block=False)
    plt.pause(1)

    #validation data plot
    plt.figure('4')
    plt.cla()
    plt.scatter(x_val,y_val)
    plt.scatter(x_val,yhatVal)
    plt.title(f'epoch={i}, ValLoss={valLoss}, w={w}, b={b}')
    plt.show(block=False)
    plt.pause(1)

    #trainLoss vs Epoch
    plt.figure('5')
    plt.cla()
    plt.plot(trainLosses)
    plt.xlabel('Epoch')
    plt.ylabel('trainLoss')
    plt.title(f'Training Loss Vs Epoch')
    plt.show(block=False)
    plt.pause(1)

    #validationLoss vs Epoch
    plt.figure('6')
    plt.cla()
    plt.plot(valLosses,color='m')
    plt.xlabel('Epoch')
    plt.ylabel('valLoss')
    plt.title(f'Validation Loss Vs Epoch')
    plt.show(block=False)
    plt.pause(1)







