import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import maximum_filter
from fact_backbone import FactNet
from common import iou
class Model(FactNet):
 def __init__(self,n,pretrained=None):
  super().__init__(n,pretrained);self.axis=nn.Conv2d(48,n*4,1);self.pixels=None;self.up1.register_forward_hook(self.capture)
 def capture(self,module,args,out):self.pixels=out
 def forward(self,x):
  out=super().forward(x);out['axis']=self.axis(self.pixels).tanh();return out
def loss(out,b):
 p=out['heat'].float().sigmoid().clamp(1e-5,1-1e-5);t=b['heat'];pos=(t==1).float();heat=(-(1-p)**2*p.log()*pos-p**2*(1-p).log()*(1-t)**4*(1-pos)).sum()/pos.sum().clamp_min(1)
 logits=out['seg'].float();t=b['seg'];p=logits.sigmoid();ce=(F.binary_cross_entropy_with_logits(logits,t,reduction='none')*(1+4*t)).mean((2,3));dice=1-(2*(p*t).sum((2,3))+1)/(p.sum((2,3))+t.sum((2,3))+1);seg=((ce+2*dice)*b['segvalid']).sum()/b['segvalid'].sum().clamp_min(1)
 B,C,H,W=out['heat'].shape;ii=torch.arange(B,device=p.device)[:,None];valid=b['valid'];den=valid.sum().clamp_min(1)
 box=out['box'].reshape(B,C,4,H*W)[ii,b['classes'],:,b['indices']];reg=(F.l1_loss(box,b['boxes'],reduction='none').mean(-1)*valid).sum()/den
 axis=out['axis'].reshape(B,C,4,H*W)[ii,b['classes'],:,b['indices']];a=F.l1_loss(axis,b['axis'],reduction='none').mean(-1);z=F.l1_loss(axis,b['axis'][:,:,[2,3,0,1]],reduction='none').mean(-1);al=(torch.minimum(a,z)*valid).sum()/den
 return heat+3*seg+3*reg+3*al
def decode(out):
 result=[]
 for hm,box,axis,seg in zip(out['heat'].float().sigmoid().cpu().numpy(),out['box'].float().cpu().numpy(),out['axis'].float().cpu().numpy(),out['seg'].float().sigmoid().cpu().numpy()):
  C,H,W=hm.shape;objects=[]
  for c in range(C):
   inds=np.flatnonzero((hm[c]==maximum_filter(hm[c],size=5))&(hm[c]>=.1));selected=[]
   for idx in sorted(inds,key=lambda k:-hm[c].ravel()[k])[:100]:
    y,x=divmod(int(idx),W);cx,cy=x/(W-1),y/(H-1);l,t,r,b=box[c*4:c*4+4,y,x];bb=[max(0,cx-float(l)),max(0,cy-float(t)),min(1,cx+float(r)),min(1,cy+float(b))];ends=axis[c*4:c*4+4,y,x].reshape(2,2)+np.array([cx,cy]);obj={'class':c,'score':float(hm[c,y,x]),'box':bb,'center':[cx,cy],'axis':ends.tolist()}
    # Same rule for every class: suppress only coincident instances, keeping overlapping elongated objects.
    if any(iou(bb,o['box'])>.7 and np.linalg.norm(np.array(o['center'])-obj['center'])<.03 for o in selected):continue
    selected.append(obj)
   objects+=selected
  result.append({'objects':objects,'seg':seg})
 return result
