# coding=utf-8
import numpy as np
import pandas as pd
import torch
from torch import nn,optim
import matplotlib.pyplot as plt
from pylab import mpl
# Solves the problem of saving the image as a box with a minus sign '-'
mpl.rcParams['axes.unicode_minus'] = False
# Normalized function of mean standard deviation
def normalize(data):
    mu =  torch.mean(data)
    std = torch.std(data)
    return (data - mu)/std
# Normalized function of max and min
def normalized_max_min(x):
	min = torch.min(x)
	max = torch.max(x)
	return (x - min) / (max - min)
# produce data
npx1=np.random.randint(0,100,(1000,1))
npx2=np.random.randint(0,100,(1000,1))
npx3=np.random.randint(0,100,(1000,1))
npx4=np.random.randint(0,100,(1000,1))
npy=(npx1*2+600)/3-(npx2+npx3)*5+npx4/3
npxy=np.concatenate((npx1,npx2,npx3,npx4,npy),axis=1)
pdxy=pd.DataFrame(npxy,columns=['x1','x2','x3','x4','y'])
# save data as excel and csv file
pdxy.to_excel('data_1.xlsx',index=False)
pdxy.to_csv('data_1.csv',index=False)
# read data from excel and csv file
pdxyexl=pd.read_excel('data_1.xlsx')
pdxycsv=pd.read_csv('data_1.csv')
# transfer data to pytorch's tensor format
data=torch.tensor(pdxyexl.values,dtype=torch.float)
train_rows=900
test_rows=100
# normalized data
xtrain=data[0:train_rows,0:-1]
xtrain=normalized_max_min(xtrain)
ytrain=data[0:train_rows,[-1]]
xtest=data[train_rows:train_rows+test_rows,0:-1]
# normalized data
xtest=normalized_max_min(xtest)
ytest=data[train_rows:train_rows+test_rows,[-1]]
#  define neural network of preceding operation
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
	        nn.Linear(4,20),
	        nn.ReLU(),
	        nn.Linear(20,10),
	        nn.ReLU(),
	        nn.Linear(10, 5),
	        nn.ReLU(),
	        nn.Linear(5,1)
        )
    def forward(self, x):
        x = self.conv1(x)
        return x
# checks whether PC has the GPU.If PC doesn't have GPU, the PC uses CPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# assign the network model to model and run the model on GPU
model=Net().to(device)
# define the type of loss function and run the loss function on GPU
criteon=nn.MSELoss().to(device)
# define optimal function and Adam is a gradient descent algorithm
# now Adam is the best algorithm in gradient descent
optimizer=optim.Adam(model.parameters(),lr=0.03)
# define the array storing each epoch's loss value when training
# one array for train, one array for test
trainlosslist=[]
testlosslist=[]
# dynamic figure
plt.ion()
# train 200 times
for epoch in range(200):
	# start training
	model.train()
	# run x and y on device(GPU)
	xtrain=xtrain.to(device)
	ytrain=ytrain.to(device)
	# The predicted value is obtained through the calculation of network model
	logits=model(xtrain)
	# The loss value is obtained through loss function
	loss=criteon(logits,ytrain)
	# clear the derivative of 'w', 'b'
	optimizer.zero_grad()
	# every time the network 'w','b' all automatically differentiate to calculate the derivative
	loss.backward()
	# update 'w' and 'b' by your definition of gradient descent's rule
	optimizer.step()
	# put training result of loss value into array
	trainlosslist.append(loss.item())
	# transfer to testing model
	model.eval()
	# test
	with torch.no_grad():
		xtest = xtest.to(device)
		ytest = ytest.to(device)
		logits = model(xtest)
		testloss = criteon(logits, ytest)
		testlosslist.append(testloss.item())
	# delete previous figure
	plt.cla()
	l1,=plt.plot(trainlosslist)
	l2,=plt.plot(testlosslist)
	plt.legend([l1,l2],['tranloss','testloss'],loc='best')
	plt.xlabel('epochs')
	# avoid drawing too fast
	plt.pause(0.2)
	plt.ioff()
plt.show()