import argparse,json,math,random,time
import cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader
from common import *
from data import Data
from model import Model,loss,decode
def components(prob,c,t,area):
 binary=np.uint8(prob[c]>t);n,lab,stats,cent=cv2.connectedComponentsWithStats(binary,8);H,W=binary.shape;out=[]
 for j in range(1,n):
  x,y,w,h,a=stats[j]
  if a<area:continue
  yy,xx=np.where(lab==j);pts=np.stack([xx/W,yy/H],1);mu=pts.mean(0);_,v=np.linalg.eigh((pts-mu).T@(pts-mu));v=v[:,-1];pr=(pts-mu)@v
  out.append({'class':c,'score':float(prob[c][lab==j].mean()),'box':[x/W,y/H,(x+w)/W,(y+h)/H],'center':[float(cent[j][0]/W),float(cent[j][1]/H)],'axis':[(mu+v*pr.min()).tolist(),(mu+v*pr.max()).tolist()]})
 return out
def objects_for(pr,settings):
 return sum([components(pr['seg'],c,s['seg_threshold'],s['min_area']) if s['mode']=='components' else [o for o in pr['objects'] if o['class']==c and o['score']>=s['threshold']] for c,s in enumerate(settings)],[])
def counts(pp,gg):
 if pp and gg:
  dist=np.linalg.norm(np.array([o['center'] for o in pp])[:,None]-np.array([o['center'] for o in gg])[None],axis=2);ii,jj=linear_sum_assignment(dist);tp=sum(dist[i,j]<.08 for i,j in zip(ii,jj))
 else:tp=0
 return tp,len(pp)-tp,len(gg)-tp
@torch.inference_mode()
def assess(net,ds):
 net.eval();pred=[];truth=[];segs=[]
 for b in DataLoader(ds,batch_size=4,shuffle=False):
  with torch.autocast('cuda',dtype=torch.float16):raw=net(b['image'].cuda())
  pred+=decode(raw)
  for k in range(len(b['image'])):
   oo=[];H,W=b['heat'].shape[-2:]
   for j in np.flatnonzero(b['valid'][k].numpy()):
    y,x=divmod(int(b['indices'][k,j]),W);oo.append({'class':int(b['classes'][k,j]),'center':[x/(W-1),y/(H-1)]})
   truth.append(oo);segs.append(b['seg'][k].numpy())
 settings=[];reports=[]
 for c in range(len(CATEGORIES[ds.cat])):
  candidates=[]
  for t in [.2,.35,.5,.65]:
   inter=sum(((p['seg'][c]>t)&(g[c]>.5)).sum() for p,g in zip(pred,segs));total=sum((p['seg'][c]>t).sum()+(g[c]>.5).sum() for p,g in zip(pred,segs));dice=float((2*inter+1)/(total+1))
   for area in [4,16,64,160]:
    tp=fp=fn=0;exact=[]
    for p,g in zip(pred,truth):
     pp=components(p['seg'],c,t,area);gg=[o for o in g if o['class']==c];a,b,z=counts(pp,gg);tp+=a;fp+=b;fn+=z;exact.append(len(pp)==len(gg))
    f1=2*tp/max(1,2*tp+fp+fn);candidates.append((.8*f1+.2*dice,float(np.mean(exact)),{'mode':'components','seg_threshold':t,'min_area':area,'point_f1':f1,'dice':dice}))
  bestseg=max(candidates,key=lambda x:(x[0],x[1]));t=bestseg[2]['seg_threshold'];dice=bestseg[2]['dice']
  for threshold in [.1,.25,.4,.55]:
   tp=fp=fn=0;exact=[]
   for p,g in zip(pred,truth):
    pp=[o for o in p['objects'] if o['class']==c and o['score']>=threshold];gg=[o for o in g if o['class']==c];a,b,z=counts(pp,gg);tp+=a;fp+=b;fn+=z;exact.append(len(pp)==len(gg))
   f1=2*tp/max(1,2*tp+fp+fn);candidates.append((.8*f1+.2*dice,float(np.mean(exact)),{'mode':'heat','threshold':threshold,'seg_threshold':t,'point_f1':f1,'dice':dice}))
  best=max(candidates,key=lambda x:(x[0],x[1]));settings.append(best[2]);reports.append({'class':CATEGORIES[ds.cat][c],'objective':best[0],'count_exact':best[1]})
 return {'objective':float(np.mean([r['objective'] for r in reports])),'settings':settings,'per_class':reports,'original_dev_sources':len(ds.rows)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--category',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 lock=read(ROOT/'training_lock.json');assert all(sha(ROOT/n)==h for n,h in lock['files'].items());assert sha(WEIGHTS)==lock['weights_sha256'];p=read(ROOT/'protocol.json');folder=ROOT/'readers'/a.category;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 ds=Data(a.category,'train',True);dd=Data(a.category,'dev');net=Model(len(CATEGORIES[a.category]),str(WEIGHTS)).cuda();enc=[p for mod in [net.stem,net.layer1,net.layer2,net.layer3] for p in mod.parameters()];ids={id(p) for p in enc};head=[p for p in net.parameters() if id(p) not in ids];opt=torch.optim.AdamW([{'params':enc,'lr':1e-4},{'params':head,'lr':1e-3}],weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');start=time.time();history=[];best=-1
 epochs=1 if a.smoke else p['training']['epochs']
 for ep in range(epochs):
  net.train()
  for mod in net.modules():
   if isinstance(mod,nn.BatchNorm2d):mod.eval()
  for q in enc:q.requires_grad_(ep>=2 or a.smoke)
  rate=.5*(1+math.cos(math.pi*ep/p['training']['epochs']));opt.param_groups[0]['lr']=1e-4*rate;opt.param_groups[1]['lr']=1e-3*rate;values=[]
  for step,b in enumerate(DataLoader(ds,batch_size=4,shuffle=False,num_workers=0)):
   b={k:v.cuda() for k,v in b.items()};opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):value=loss(net(b['image']),b)
   assert torch.isfinite(value);scaler.scale(value).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(net.parameters(),5);scaler.step(opt);scaler.update();values.append(float(value.detach()))
   if a.smoke and step==1:break
  row={'epoch':ep+1,'loss':float(np.mean(values)),'seconds':time.time()-start}
  if a.smoke:dump(folder/'smoke.json',{'state':'passed','category':a.category,'row':row,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,'heads':[k for k in net(b['image']).keys()]});return
  if ep==0 or (ep+1)%4==0:
   metric=assess(net,dd);row['dev']=metric
   if metric['objective']>best:
    best=metric['objective'];torch.save({'model':{k:v.cpu() for k,v in net.state_dict().items()},'dev':json.loads(json.dumps(metric)),'epoch':ep+1,'train_parent_ids':[r['source_id'] for r in ds.rows]},folder/'best.pt')
  history.append(row);dump(folder/'history.json',history);dump(folder/'status.json',row);print(a.category,json.dumps(row),flush=True)
 ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);dump(folder/'complete.json',{'state':'complete','category':a.category,'epoch':ck['epoch'],'checkpoint_sha256':sha(folder/'best.pt'),'dev':ck['dev'],'train_parent_ids':[r['source_id'] for r in ds.rows],'dev_parent_ids':[r['source_id'] for r in dd.rows],'initialization':'ImageNet only','seconds':time.time()-start,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30})
if __name__=='__main__':main()
