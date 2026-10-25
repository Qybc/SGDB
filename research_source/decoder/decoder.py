"""Object-set decoding with optional mask feedback and edit-equivariant training."""
import copy,math,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from io_utils import V1
sys.path.insert(0,str(V1))
from common import CATEGORIES
SOURCES=64
FREE=64
SIZE=(80,112)

def query_geometry(objects,device,duplicate=False):
    # Slots do not encode expected normal counts. Free slots remain available.
    selected=sorted(enumerate(objects),key=lambda z:(-z[1].get('score',1),z[0]))[:SOURCES]
    oo=[copy.deepcopy(o) for _,o in selected]
    if duplicate and oo:
        oo += [copy.deepcopy(o) for o in oo[:min(8,SOURCES-len(oo))]]
    geom=[];classes=[];confidence=[];padding=[]
    for o in oo:
        geom.append(o['center']+o['box']+np.asarray(o['axis']).ravel().tolist())
        classes.append(o['class']);confidence.append(o.get('score',.5));padding.append(False)
    while len(geom)<SOURCES:
        geom.append([.5,.5,.4,.4,.6,.6,.4,.5,.6,.5]);classes.append(-1);confidence.append(0);padding.append(True)
    for y in range(8):
        for x in range(8):
            cx,cy=(x+.5)/8,(y+.5)/8
            geom.append([cx,cy,max(0,cx-.125),max(0,cy-.125),min(1,cx+.125),min(1,cy+.125),max(0,cx-.1),cy,min(1,cx+.1),cy])
            classes.append(-1);confidence.append(0);padding.append(False)
    return (torch.tensor(geom,dtype=torch.float32,device=device),torch.tensor(classes,device=device),
            torch.tensor(confidence,dtype=torch.float32,device=device),torch.tensor(padding,device=device))

class Layer(nn.Module):
    def __init__(self,d=64):
        super().__init__();self.selfatt=nn.MultiheadAttention(d,4,dropout=0,batch_first=True)
        self.cross=nn.MultiheadAttention(d,4,dropout=0,batch_first=True)
        self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.ff=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,d))
    def forward(self,q,memory,padding,bias):
        z=self.norm[0](q);q=q+self.selfatt(z,z,z,key_padding_mask=padding,need_weights=False)[0]
        z=self.norm[1](q);q=q+self.cross(z,memory,memory,attn_mask=bias.repeat_interleave(4,0),need_weights=False)[0]
        return q+self.ff(self.norm[2](q))

class FactDecoder(nn.Module):
    def __init__(self,classes,variant):
        super().__init__();self.classes=classes;self.variant=variant
        self.pixel=nn.Sequential(nn.Conv2d(48,64,1),nn.GroupNorm(8,64),nn.GELU())
        self.position=nn.Linear(2,64);self.geom=nn.Linear(10,64);self.label=nn.Embedding(classes+1,64)
        self.slots=nn.Parameter(torch.randn(SOURCES+FREE,64)*.02)
        self.layers=nn.ModuleList([Layer(),Layer()]);self.mask=nn.Linear(64,64)
        self.classifier=nn.Linear(64,classes+1);self.geometry=nn.Linear(64,10)
        nn.init.zeros_(self.classifier.weight);nn.init.zeros_(self.classifier.bias)
        nn.init.zeros_(self.geometry.weight);nn.init.zeros_(self.geometry.bias)
    def forward(self,parents,duplicate=False):
        device=next(self.parameters()).device
        features=torch.stack([p['features'] for p in parents]).to(device).float()
        pixel=self.pixel(features);B,D,H,W=pixel.shape
        y,x=torch.meshgrid(torch.linspace(0,1,H,device=device),torch.linspace(0,1,W,device=device),indexing='ij')
        xy=torch.stack([x,y],-1)
        mem=F.adaptive_avg_pool2d(pixel,(40,56));my,mx=torch.meshgrid(torch.linspace(0,1,40,device=device),torch.linspace(0,1,56,device=device),indexing='ij')
        mxy=torch.stack([mx,my],-1).reshape(-1,2)
        memory=mem.flatten(2).transpose(1,2)+self.position(mxy)[None]
        data=[query_geometry(p['anchor'],device,duplicate) for p in parents]
        geom,labels,confidence,padding=[torch.stack([r[i] for r in data]) for i in range(4)]
        grid=geom[:,:,:2,None].transpose(2,3).reshape(B,1,-1,2)*2-1
        sampled=F.grid_sample(pixel,grid,align_corners=True)[:,:,0].transpose(1,2)
        q=self.slots[None]+self.geom(geom)+self.label(labels.clamp_min(0).where(labels>=0,torch.full_like(labels,self.classes)))+sampled
        wh=(geom[:,:,4:6]-geom[:,:,2:4]).clamp_min(.04)
        bias=-.5*(((mxy[None,None]-geom[:,:,:2,None].transpose(2,3))/wh[:,:,None])**2).sum(-1)
        bias=bias.clamp(-8,0)
        prior=torch.zeros(B,SOURCES+FREE,self.classes+1,device=device)
        prior[:,:,self.classes]=2
        for b in range(B):
            ids=torch.where(labels[b]>=0)[0]
            prior[b,ids,self.classes]=0
            prior[b,ids,labels[b,ids]]=torch.logit(confidence[b,ids].clamp(.1,.95))+1
        masks=None
        for index,layer in enumerate(self.layers):
            if index and self.variant=='evidence_edit':
                mb=F.interpolate(masks,size=(40,56),mode='bilinear',align_corners=False).flatten(2)
                use_bias=(bias+F.logsigmoid(mb)).clamp(-12,0)
            else:use_bias=bias
            q=layer(q,memory,padding,use_bias)
            masks=torch.einsum('bnd,bdhw->bnhw',self.mask(q),pixel)/8
            # Soft spatial support; no hard foreground threshold enters attention.
            spatial=-.5*(((xy[None,None]-geom[:,:,:2,None,None].permute(0,1,3,4,2))/wh[:,:,None,None])**2).sum(-1)
            masks=masks+spatial.clamp(-12,0)
        delta=self.geometry(q).tanh()
        scale=torch.cat([wh,wh.repeat(1,1,2),wh.repeat(1,1,2)],-1)*.5
        pred=(geom+delta*scale).clamp(0,1)
        lo=torch.minimum(pred[:,:,2:4],pred[:,:,4:6]);hi=torch.maximum(pred[:,:,2:4],pred[:,:,4:6])
        pred=torch.cat([pred[:,:,:2],lo,hi,pred[:,:,6:]],-1)
        logits=self.classifier(q)+prior
        logits=logits.masked_fill(padding[:,:,None],-1e4)
        logits[:,:,self.classes]=torch.where(padding,torch.zeros_like(logits[:,:,self.classes]),logits[:,:,self.classes])
        return {'logits':logits,'geometry':pred,'masks':masks,'padding':padding}

def match(out,b,p):
    n=len(p['truth'])
    if not n:return np.array([],int),np.array([],int)
    device=out['logits'].device
    g=torch.tensor([o['center']+o['box'] for o in p['truth']],device=device)
    labels=torch.tensor([o['class'] for o in p['truth']],device=device)
    prob=out['logits'][b].softmax(-1)
    cost=-prob[:,labels].log().clamp_min(-20)+5*torch.cdist(out['geometry'][b,:,:2],g[:,:2],p=1)+2*torch.cdist(out['geometry'][b,:,2:6],g[:,2:],p=1)
    pp=F.interpolate(out['masks'][b,:,None],size=(20,28),mode='bilinear',align_corners=False)[:,0].sigmoid().flatten(1)
    tt=F.interpolate(p['masks'].to(device).float()[:,None],size=(20,28),mode='nearest')[:,0].flatten(1)
    dice=1-(2*(pp@tt.T)+1)/(pp.sum(1)[:,None]+tt.sum(1)[None]+1)
    valid=tt.sum(1)>0;cost=cost+.5*dice*valid[None]
    cost=cost.masked_fill(out['padding'][b,:,None],1e6)
    return linear_sum_assignment(cost.detach().cpu().numpy())

def loss(out,parents,variant):
    device=out['logits'].device;zero=out['geometry'].sum()*0
    terms={k:zero for k in ['class','center','box','axis','mask','binding','edit']};matches=[]
    for b,p in enumerate(parents):
        ii,jj=match(out,b,p);matches.append((ii,jj))
        target=torch.full((SOURCES+FREE,),out['logits'].shape[-1]-1,device=device,dtype=torch.long)
        if len(ii):target[ii]=torch.tensor([p['truth'][j]['class'] for j in jj],device=device)
        weight=torch.ones(out['logits'].shape[-1],device=device);weight[-1]=.2
        ce=F.cross_entropy(out['logits'][b],target,weight=weight,reduction='none')
        active=~out['padding'][b];terms['class']=terms['class']+ce[active].mean()
        if not len(ii):continue
        truth=torch.tensor([p['truth'][j]['center']+p['truth'][j]['box']+np.asarray(p['truth'][j]['axis']).ravel().tolist() for j in jj],device=device)
        pred=out['geometry'][b,ii]
        terms['center']=terms['center']+F.l1_loss(pred[:,:2],truth[:,:2])
        terms['box']=terms['box']+F.l1_loss(pred[:,2:6],truth[:,2:6])
        ae=(pred[:,6:]-truth[:,6:]).abs().mean(1);be=(pred[:,6:]-truth[:,[8,9,6,7]]).abs().mean(1)
        terms['axis']=terms['axis']+torch.minimum(ae,be).mean()
        tt=p['masks'][jj].to(device).float();valid=tt.sum((1,2))>=4
        if valid.any():
            ml=out['masks'][b,ii][valid];gt=tt[valid];mp=ml.sigmoid()
            terms['mask']=terms['mask']+F.binary_cross_entropy_with_logits(ml,gt)+(1-(2*(mp*gt).sum((1,2))+1)/(mp.sum((1,2))+gt.sum((1,2))+1)).mean()
            if variant=='evidence_edit':
                yy,xx=torch.meshgrid(torch.linspace(0,1,SIZE[0],device=device),torch.linspace(0,1,SIZE[1],device=device),indexing='ij')
                xy=torch.stack([xx,yy],-1);pc=(mp[:,:,:,None]*xy).sum((1,2))/mp.sum((1,2))[:,None].clamp_min(1e-5)
                gc=(gt[:,:,:,None]*xy).sum((1,2))/gt.sum((1,2))[:,None].clamp_min(1)
                terms['binding']=terms['binding']+F.smooth_l1_loss(pred[valid,:2]-pc,truth[valid,:2]-gc,beta=.02)
                native=torch.tensor(p['native_size'],device=device,dtype=torch.float32);native=native/native.norm()
                a=pred[valid,6:8]*native;z=pred[valid,8:10]*native;v=z-a;v=v/v.norm(dim=1,keepdim=True).clamp_min(.01)
                points=xy[None]*native;dist=((points-a[:,None,None])*torch.stack([-v[:,1],v[:,0]],1)[:,None,None]).sum(-1).square()
                ga=truth[valid,6:8]*native;gz=truth[valid,8:10]*native;gv=gz-ga;gv=gv/gv.norm(dim=1,keepdim=True).clamp_min(.01)
                gd=((points-ga[:,None,None])*torch.stack([-gv[:,1],gv[:,0]],1)[:,None,None]).sum(-1).square()
                terms['binding']=terms['binding']+((mp*dist).sum((1,2))/mp.sum((1,2)).clamp_min(1)-(gt*gd).sum((1,2))/gt.sum((1,2)).clamp_min(1)).abs().mean()
    for k in terms:terms[k]=terms[k]/len(parents)
    if variant=='evidence_edit' and len(parents)==2 and parents[0]['source_id']==parents[1]['source_id']:
        c=out['logits'].shape[-1]-1;counts=out['logits'].softmax(-1)[:,:,:c].sum(1)
        actual=torch.tensor([[sum(o['class']==k for o in p['truth']) for k in range(c)] for p in parents],device=device)
        terms['edit']=F.smooth_l1_loss(counts[1]-counts[0],actual[1]-actual[0],beta=1)
        maps=[{p['truth'][j]['uid']:int(i) for i,j in zip(ii,jj)} for p,(ii,jj) in zip(parents,matches)]
        shared=sorted(set(maps[0])&set(maps[1]))
        if shared:
            a=torch.tensor([maps[0][u] for u in shared],device=device);z=torch.tensor([maps[1][u] for u in shared],device=device)
            ta={o['uid']:o for o in parents[0]['truth']};tb={o['uid']:o for o in parents[1]['truth']}
            delta=torch.tensor([np.array(tb[u]['center'])-ta[u]['center'] for u in shared],device=device,dtype=torch.float32)
            terms['edit']=terms['edit']+F.smooth_l1_loss(out['geometry'][1,z,:2]-out['geometry'][0,a,:2],delta,beta=.02)
    total=terms['class']+2*terms['center']+2*terms['box']+3*terms['axis']+terms['mask']+.25*terms['binding']+.25*terms['edit']
    return total,{k:float(v.detach()) for k,v in terms.items()}

def decoded(out,b):
    probs=out['logits'][b].softmax(-1).detach().cpu();labels=probs.argmax(-1);geom=out['geometry'][b].detach().cpu().tolist();objects=[];slots=[]
    for i,c in enumerate(labels.tolist()):
        if c==probs.shape[-1]-1 or bool(out['padding'][b,i]):continue
        v=geom[i];objects.append({'class':c,'score':float(probs[i,c]),'center':v[:2],'box':v[2:6],'axis':[v[6:8],v[8:10]]});slots.append(i)
    order=sorted(range(len(objects)),key=lambda i:(objects[i]['class'],-objects[i]['score'],slots[i]))
    return [objects[i] for i in order],[slots[i] for i in order]
