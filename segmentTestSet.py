import sys
sys.path.append("../")
import numpy as np
import torch
from net_v1 import UNet3d
import os
from NetworkTraining_py.forwardOnBigImages import processBigInput
from NetworkTraining_py.crop import crop
from skimage.external.tifffile import imread

datadir="../WormData"
exec(open(os.path.join(datadir,"testFiles_AIY.txt")).read())

log_dir="log_v1"
net = UNet3d().cuda()
saved_net=torch.load(os.path.join(log_dir,"net_Test_bestF1.pth")) #
net.load_state_dict(saved_net['state_dict'])
net.eval();

out_dir="output_best_v1" 

def process_output(o):
    e=np.exp(o[0,1,:,:,:])
    prob=e/(e+1)
    return prob
  
outdir=os.path.join(log_dir,out_dir)
os.makedirs(outdir)

cropsz=np.array([80,80,80])

for f in testFiles: 
  img=imread(os.path.join(datadir,f[0])).astype(np.float32)
  # this `cropping' is in fact enlarging the volume to at least 80 in each dim
  cropinds=[]
  uncropinds=[]
  for i in range(img.ndim):
    marg=cropsz[i]-img.shape[i]
    if marg>0:
      marg1=marg//2
      marg2=marg-marg1
      cropinds.append(slice(-marg1,img.shape[i]+marg2))
      uncropinds.append(slice(marg1,img.shape[i]+marg1))
    else:
      cropinds.append(slice(0,img.shape[i]))
      uncropinds.append(slice(0,img.shape[i]))
  imgc,_=crop(img,cropinds,fill=0)
  inp=imgc.reshape(1,1,imgc.shape[-3],imgc.shape[-2],imgc.shape[-1])
  oup=processBigInput(inp,cropsz,(22,22,22),2,net,(1,2,))
  prob=process_output(oup)
  probc,_=crop(prob,uncropinds)
  np.save(os.path.join(outdir,os.path.basename(f[0])),probc)
