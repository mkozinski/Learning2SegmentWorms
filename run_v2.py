import sys
sys.path.append("../")
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from net_v1 import UNet3d
from NetworkTraining_py.loggerBasic import LoggerBasic
from NetworkTraining_py.loggerF1 import LoggerF1
from NetworkTraining_py.loggerComposit import LoggerComposit
from NetworkTraining_py.dataset import Dataset
from NetworkTraining_py.datasetCrops import TestDataset
from NetworkTraining_py.crop import crop
import os
import os.path
import torch.nn.functional as F
from shutil import copyfile
from NetworkTraining_py.trainer import trainer
from NetworkTraining_py.tester import tester
from random import randint, random
import math
from skimage.external.tifffile import imread

def countClasses(lbls):
  count=np.array([0,0])
  for lbl in lbls:
    count[0]+=np.equal(lbl,0).sum()
    count[1]+=np.equal(lbl,1).sum()
  return count

def calcClassWeights(count):
  # reweight gently - do not balance completely
  freq=count.astype(np.double)/count.sum()
  freq+=1.02
  freq=np.log(freq)
  w=np.power(freq,-1)
  w=w/w.sum()
  return torch.Tensor(w)

def augmentTrain(img,lbl,cropSz):

  # random crop 
  njitter=8
  maxstartind1=lbl.shape[0]-cropSz[0]
  maxstartind2=lbl.shape[1]-cropSz[1]
  maxstartind3=lbl.shape[2]-cropSz[2]
  if maxstartind1>0:
    startind1=randint(0,maxstartind1)
  else:
    startind1=maxstartind1-njitter//2+randint(0,njitter)
  if maxstartind2>0:
    startind2=randint(0,maxstartind2)
  else:
    startind2=maxstartind2-njitter//2+randint(0,njitter)
  if maxstartind3>0:
    startind3=randint(0,maxstartind3)
  else:
    startind3=maxstartind3-njitter//2+randint(0,njitter)
  cropinds=[slice(startind1,startind1+cropSz[0]),
            slice(startind2,startind2+cropSz[1]),
            slice(startind3,startind3+cropSz[2])]
  img,_=crop(img,cropinds)
  lbl,_=crop(lbl,cropinds,fill=255)

  # flip
  if random()>0.5 :
    img=np.flip(img,0)
    lbl=np.flip(lbl,0)
  if random()>0.5 :
    img=np.flip(img,1)
    lbl=np.flip(lbl,1)
  if random()>0.5 :
    img=np.flip(img,2)
    lbl=np.flip(lbl,2)

  # copy the np array to avoid negative strides resulting from flip
  it=torch.from_numpy(np.copy(img)).unsqueeze(0)
  lt=torch.from_numpy(lbl.astype(np.long))

  return it,lt

def preproc(img,lbl):
  # this function is used in the training loop, and not in the data loader
  # this is needed to avoid transfer of data to GPU in threads of the dataloader
  return img.cuda(), lbl.cuda()

def augmentTest(img,lbl):
  lbl=torch.from_numpy(lbl)
  img=torch.from_numpy(img).unsqueeze(0)
  return img,lbl

def test_preproc(output,target):
  # preprocessing for test logger
  idx=torch.Tensor.mul_(target<=1, target>=0).reshape(target.numel())
  o=output[:,1,:,:,:]
  oo=o.reshape(target.numel())[idx]
  t=target.reshape(target.numel())[idx]
  o=torch.pow(torch.exp(-oo)+1,-1)
  return o,t

def addMarginToTestVol(v,fill=0):
  cropsz=np.array([80,80,80])
  cropinds=[]
  for i in range(v.ndim):
    marg=cropsz[i]-v.shape[i]
    if marg>0:
      marg1=marg//2
      marg2=marg-marg1
      cropinds.append(slice(-marg1,v.shape[i]+marg2))
    else:
      cropinds.append(slice(0,v.shape[i]))
  c,_=crop(v,cropinds,fill=fill)
  return c

log_dir="log_v1"
datadir="../WormData/"
exec(open("../WormData/trainFiles_AIY.txt").read())
exec(open("../WormData/testFiles_AIY.txt").read())
trainimgs=[]
trainlbls=[]
for f in trainFiles:
  img =imread (os.path.join(datadir,f[0]))
  lbl =np.load(os.path.join(datadir,f[1])).astype(np.uint8)
  trainimgs.append(img.astype(np.float32))
  trainlbls.append(lbl)
testimgs=[]
testlbls=[]
for f in testFiles:
  img  =imread (os.path.join(datadir,f[0]))
  print(img.shape)
  img  =addMarginToTestVol(img)
  print(img.shape)
  lbl  =np.load(os.path.join(datadir,f[1]))
  lbl  =addMarginToTestVol(lbl,fill=255)
  testimgs.append(img.astype(np.float32))
  testlbls.append(lbl)

os.makedirs(log_dir)
copyfile(__file__,os.path.join(log_dir,"setup.txt"))

train_dataset=Dataset(trainimgs,trainlbls,
                 lambda i,l: augmentTrain(i,l,np.array([48,48,48])))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                  shuffle=True, num_workers=16, drop_last=True)

test_dataset = TestDataset(testimgs,testlbls,np.array([80,80,80]),
                  [22,22,22], augmentTest, ignoreInd=255)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                  shuffle=False, num_workers=1, drop_last=False)

net = UNet3d().cuda()
#prev_log_dir="log_v2"
#saved_net=torch.load(os.path.join(prev_log_dir,"net_last.pth"))
#net.load_state_dict(saved_net['state_dict'])
net.train()

weight=calcClassWeights(countClasses(train_dataset.lbl))
loss = torch.nn.CrossEntropyLoss(weight=weight,ignore_index=255).cuda()
print("loss.weight",loss.weight)

optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)

logger= LoggerBasic(log_dir,"Basic",100)

logger_test=LoggerF1(log_dir,"Test",test_preproc, nBins=10000, saveBest=True)
tstr=tester(test_loader,logger_test,preproc)

lr_lambda=lambda e: 1/(1+e*1e-5)
lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
trn=trainer(net, train_loader, optimizer, loss, logger, tstr, 500,
            lr_scheduler=lr_scheduler,preprocImgLbl=preproc)

if __name__ == '__main__':
  print(log_dir)
  trn.train(100000)
