import numpy as np
from scipy.ndimage import maximum_filter,label,find_objects
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import wide_resnet50_2
from common import iou

class Block(nn.Sequential):
    def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.ReLU())
class FactNet(nn.Module):
    def __init__(self,n,pretrained=None):
        super().__init__();b=wide_resnet50_2(weights=None)
        if pretrained:b.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True))
        self.n=n;self.stem=nn.Sequential(b.conv1,b.bn1,b.relu);self.pool=b.maxpool;self.layer1=b.layer1;self.layer2=b.layer2;self.layer3=b.layer3
        self.context=nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024*16,256),nn.ReLU(),nn.Dropout(.1));self.condition=nn.Linear(256,256)
        self.up3=Block(1024+512,128);self.up2=Block(128+256,96);self.up1=Block(96+64+2,48);self.heat=nn.Conv2d(48,n,1);self.box=nn.Conv2d(48,n*4,1);self.seg=nn.Conv2d(48,n,1);nn.init.constant_(self.heat.bias,-2.19)
    def forward(self,x):
        s=self.stem(x);a=self.layer1(self.pool(s));b=self.layer2(a);c=self.layer3(b);z=self.context(c);u=self.up3(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));scale,bias=self.condition(z).chunk(2,1);u=u*(1+.1*scale.tanh()[:,:,None,None])+.1*bias[:,:,None,None];u=self.up2(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));yy,xx=torch.meshgrid(torch.linspace(-1,1,s.shape[-2],device=x.device,dtype=s.dtype),torch.linspace(-1,1,s.shape[-1],device=x.device,dtype=s.dtype),indexing='ij');xy=torch.stack([xx,yy])[None].expand(x.shape[0],-1,-1,-1);u=self.up1(torch.cat([F.interpolate(u,size=s.shape[-2:],mode='bilinear',align_corners=False),s,xy],1));return {'heat':self.heat(u),'box':self.box(u).sigmoid(),'seg':self.seg(u)}
def loss_for(out,b):
    p=out['heat'].float().sigmoid().clamp(1e-5,1-1e-5);t=b['heat'];pos=(t==1).float();neg=1-pos;heat=(-(1-p)**2*p.log()*pos-p**2*(1-p).log()*(1-t)**4*neg).sum()/pos.sum().clamp_min(1)
    p=out['seg'].sigmoid();t=b['seg'];seg=F.binary_cross_entropy_with_logits(out['seg'],t)+(1-(2*(p*t).sum((2,3))+1)/(p.sum((2,3))+t.sum((2,3))+1)).mean();B,C,H,W=out['heat'].shape;bb=torch.arange(B,device=p.device)[:,None];v=out['box'].reshape(B,C,4,H*W)[bb,b['classes'],:,b['indices']];reg=(F.l1_loss(v,b['boxes'],reduction='none').mean(-1)*b['valid']).sum()/b['valid'].sum().clamp_min(1);return heat+seg+5*reg
def decode(out,min_score=.1):
    result=[]
    for hm,boxes,seg in zip(out['heat'].float().sigmoid().cpu().numpy(),out['box'].float().cpu().numpy(),out['seg'].float().sigmoid().cpu().numpy()):
        C,H,W=hm.shape;objects=[]
        for c in range(C):
            inds=np.flatnonzero((hm[c]==maximum_filter(hm[c],size=5))&(hm[c]>=min_score));chosen=[]
            for idx in sorted(inds,key=lambda z:-hm[c].ravel()[z])[:100]:
                y,x=divmod(int(idx),W);cx,cy=x/(W-1),y/(H-1);l,t,r,b=boxes[c*4:c*4+4,y,x];bb=[max(0,cx-float(l)),max(0,cy-float(t)),min(1,cx+float(r)),min(1,cy+float(b))]
                if any(iou(bb,o['box'])>.7 for o in chosen):continue
                chosen.append({'class':c,'score':float(hm[c,y,x]),'center':[cx,cy],'box':bb})
            objects+=chosen
        result.append({'objects':objects,'seg':seg})
    return result
def components(seg,c,min_area):
    a,n=label(seg[c]>.5);objects=[];H,W=a.shape
    for j,sl in enumerate(find_objects(a),1):
        if sl is None:continue
        mask=a[sl]==j;area=mask.sum()
        if area<min_area:continue
        y,x=np.where(mask);y=y+sl[0].start;x=x+sl[1].start;objects.append({'class':c,'score':float(seg[c][a==j].mean()),'center':[float((x.min()+x.max())/(2*W)),float((y.min()+y.max())/(2*H))],'box':[float(x.min()/W),float(y.min()/H),float((x.max()+1)/W),float((y.max()+1)/H)]})
    return objects
def read_objects(row,settings):
    objects=[]
    for c,s in enumerate(settings):
        if s['mode']=='components':objects+=components(row['seg'],c,s['min_area'])
        else:objects += [o for o in row['objects'] if o['class']==c and o['score']>=s['threshold']]
    return objects
