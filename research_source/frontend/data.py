import copy,random
import cv2,numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset
from common import *
cv2.setNumThreads(1)
def preprocess(rgb):
 rgb=cv2.resize(rgb,SIZE,interpolation=cv2.INTER_AREA);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(2.,(8,8)).apply(lab[:,:,0]);rgb=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
 a=(rgb.astype('float32')/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32);return torch.from_numpy(a.transpose(2,0,1).copy())
def edit(rgb,m,objects,rng,mode):
 candidates=[i for i,a in enumerate(m) if a.sum()>8 and .002<a.mean()<.3]
 if not candidates:return rgb,m,objects
 j=int(rng.choice(candidates));own=m[j]>0;other=np.maximum.reduce(np.delete(m,j,0),axis=0)>0 if len(m)>1 else np.zeros_like(own)
 # Do not inpaint/paste over another labelled instance or semantic enclosure.
 if (own&other).sum()>.02*own.sum():return rgb,m,objects
 yy,xx=np.where(own);x0,x1,y0,y1=xx.min(),xx.max()+1,yy.min(),yy.max()+1;w=x1-x0;h=y1-y0;patch=rgb[y0:y1,x0:x1].copy();pm=own[y0:y1,x0:x1]
 if mode=='remove':
  rgb=cv2.inpaint(rgb,np.uint8(own)*255,3,cv2.INPAINT_TELEA);keep=[i for i in range(len(m)) if i!=j];return rgb,m[keep],[objects[i] for i in keep]
 for _ in range(20):
  nx=int(rng.integers(0,513-w));ny=int(rng.integers(0,513-h))
  if np.maximum.reduce(m)[ny:ny+h,nx:nx+w][pm].sum()>0:continue
  if mode=='move':rgb=cv2.inpaint(rgb,np.uint8(own)*255,3,cv2.INPAINT_TELEA)
  rgb[ny:ny+h,nx:nx+w][pm]=patch[pm];new=np.zeros_like(m[j]);new[ny:ny+h,nx:nx+w]=pm;o=copy.deepcopy(objects[j]);dx,dy=(nx-x0)/512,(ny-y0)/512;o['box']=[o['box'][0]+dx,o['box'][1]+dy,o['box'][2]+dx,o['box'][3]+dy];o['center']=[o['center'][0]+dx,o['center'][1]+dy];o['axis']=[[x+dx,y+dy] for x,y in o['axis']]
  if mode=='move':m[j]=new;objects[j]=o
  else:m=np.concatenate([m,new[None]]);objects.append(o)
  break
 return rgb,m,objects
class Data(Dataset):
 def __init__(self,cat,split,training=False):
  self.cat=cat;self.training=training;self.rows=[r for r in read(ROOT/'training_records.json') if r['category']==cat and r['split']==split];self.cache=[]
  for r in self.rows:
   rgb=np.array(Image.open(r['image_path']).convert('RGB').resize((512,512),Image.Resampling.LANCZOS));z=np.load(r['mask_path']);m=np.stack([np.array(Image.fromarray(a).resize((512,512),Image.Resampling.NEAREST)) for a in z['masks']]);self.cache.append((rgb,m,r['objects']))
  assert self.rows
 def __len__(self):return 240 if self.training else len(self.rows)*2
 def __getitem__(self,index):
  idx=random.randrange(len(self.rows)) if self.training else index//2;rng=np.random.default_rng(random.randrange(2**32) if self.training else int(hashlib.sha256((self.rows[idx]['source_id']+str(index%2)).encode()).hexdigest()[:12],16));rgb,m,oo=self.cache[idx];rgb=rgb.copy();m=m.copy();oo=copy.deepcopy(oo)
  if (self.training and rng.random()<.6) or (not self.training and index%2):rgb,m,oo=edit(rgb,m,oo,rng,str(rng.choice(['move','add','remove'])))
  if self.training:
   for dim in [0,1]:
    if rng.random()<.5:
     rgb=np.flip(rgb,dim).copy();m=np.flip(m,dim+1).copy()
     for o in oo:
      if dim==1:a,b,c,d=o['box'];o['box']=[1-c,b,1-a,d];o['center'][0]=1-o['center'][0];o['axis']=[[1-x,y] for x,y in o['axis']]
      else:a,b,c,d=o['box'];o['box']=[a,1-d,c,1-b];o['center'][1]=1-o['center'][1];o['axis']=[[x,1-y] for x,y in o['axis']]
   x=rgb.astype('float32')/255;x=np.clip(x*float(rng.uniform(.5,1.5)),0,1)**float(rng.uniform(.7,1.5));rgb=np.uint8(x*255)
   if rng.random()<.2:rgb=cv2.GaussianBlur(rgb,(3,3),0)
  C=len(CATEGORIES[self.cat]);W,H=OUT;yy,xx=np.mgrid[:H,:W];heat=np.zeros((C,H,W),np.float32);seg=heat.copy();indices=np.zeros(128,np.int64);classes=indices.copy();boxes=np.zeros((128,4),np.float32);axes=boxes.copy();valid=np.zeros(128,np.float32);sv=np.ones(C,np.float32)
  assert len(oo)<=128
  for j,(o,mask) in enumerate(zip(oo,m)):
   c=o['class'];cx,cy=o['center'];x=int(np.clip(round(cx*(W-1)),0,W-1));y=int(np.clip(round(cy*(H-1)),0,H-1));px,py=x/(W-1),y/(H-1);a,b,d,e=o['box'];rad=max(1.2,min(4.,min(d-a,e-b)*H*.12));heat[c]=np.maximum(heat[c],np.exp(-((xx-x)**2+(yy-y)**2)/(2*rad**2)));seg[c]=np.maximum(seg[c],cv2.resize(mask,OUT,interpolation=cv2.INTER_NEAREST));indices[j]=y*W+x;classes[j]=c;boxes[j]=[max(0,px-a),max(0,py-b),max(0,d-px),max(0,e-py)];axes[j]=(np.array(o['axis'])-np.array([px,py])).ravel();valid[j]=1
   if mask.sum()<8:sv[c]=0
  result={k:torch.from_numpy(v) for k,v in {'heat':heat,'seg':seg,'indices':indices,'classes':classes,'boxes':boxes,'axis':axes,'valid':valid,'segvalid':sv}.items()};result['image']=preprocess(rgb);return result
