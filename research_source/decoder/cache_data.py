import copy,hashlib,sys,time
import cv2,numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from io_utils import ROOT,V1,read,rows,sha,dump,save
sys.path.insert(0,str(V1))
from common import CATEGORIES
from data import preprocess,edit
from model import Model,decode
from train import objects_for
from decoder import SIZE

def network(cat):
    path=V1/'readers'/cat/'best.pt';digest=sha(path)
    assert digest==read(path.parent/'complete.json')['checkpoint_sha256']
    ck=torch.load(path,map_location='cpu',weights_only=True);net=Model(len(CATEGORIES[cat])).cuda().eval().requires_grad_(False);net.load_state_dict(ck['model'])
    return net,ck['dev']['settings'],digest

@torch.inference_mode()
def extract(net,rgb,settings):
    with torch.autocast('cuda',dtype=torch.float16):out=net(preprocess(rgb)[None].cuda())
    feat=F.adaptive_avg_pool2d(net.pixels.float(),SIZE)[0].cpu().half().clone()
    pr=decode(out)[0];anchor=objects_for(pr,settings)
    return feat,anchor,out['seg'][0].sigmoid().float().cpu()

def cache(cat):
    folder=ROOT/'readers'/cat;folder.mkdir(parents=True,exist_ok=True);done=folder/'cache_complete.json';path=folder/'normal_cache.pt'
    if done.exists():
        r=read(done);assert sha(path)==r['sha256'];return torch.load(path,map_location='cpu',weights_only=False),r
    records=[r for r in read(V1/'training_records.json') if r['category']==cat];assert records and all(r['split'] in ['train','dev'] for r in records)
    normal={r['source_id']:r for r in rows(V1/'readers'/cat/'normal_facts.jsonl') if r['split'] in ['train','dev']}
    net,settings,digest=network(cat);parents=[];started=time.time();mask_hashes={};edit_counts={}
    for index,row in enumerate(records):
        assert sha(row['image_path'])==row['sha256'];im=Image.open(row['image_path']).convert('RGB');rgb=np.array(im);size=list(im.size)
        z=np.load(row['mask_path'])['masks'];mask_hashes[row['mask_path']]=sha(row['mask_path'])
        truth=[{**{k:copy.deepcopy(o[k]) for k in ['class','center','box','axis']},'uid':str(j)} for j,o in enumerate(row['objects'])]
        feat,_,_=extract(net,rgb,settings)
        assert normal[row['source_id']]['sha256']==row['sha256'] and normal[row['source_id']]['split']==row['split']
        base={'source_id':row['source_id'],'sha256':row['sha256'],'split':row['split'],'dataset':row['dataset'],'category':cat,'native_size':size}
        masks=np.stack([cv2.resize(np.uint8(m),SIZE[::-1],interpolation=cv2.INTER_NEAREST) for m in z])
        parents.append({**base,'view':'original','edit':'none','features':feat,'anchor':normal[row['source_id']]['objects'],'truth':truth,'masks':torch.from_numpy(masks)})
        seed=int(hashlib.sha256((row['source_id']+'DFD_EDIT_V1').encode()).hexdigest()[:12],16);rng=np.random.default_rng(seed)
        mode=['add','remove','move'][seed%3]
        ergb=np.array(im.resize((512,512),Image.Resampling.LANCZOS));em=np.stack([cv2.resize(np.uint8(m),(512,512),interpolation=cv2.INTER_NEAREST) for m in z])
        ergb,em,eo=edit(ergb,em,copy.deepcopy(truth),rng,mode)
        if mode=='add' and len(eo)>len(truth):eo[-1]['uid']='added_'+eo[-1]['uid']
        effective=eo!=truth;name=mode if effective else 'identity_edit_rejected'
        ef,ea,_=extract(net,ergb,settings);mm=np.stack([cv2.resize(np.uint8(m),SIZE[::-1],interpolation=cv2.INTER_NEAREST) for m in em]) if len(em) else np.zeros((0,*SIZE),np.uint8)
        parents.append({**base,'view':'edited','edit':name,'features':ef,'anchor':ea,'truth':eo,'masks':torch.from_numpy(mm)})
        edit_counts[name]=edit_counts.get(name,0)+1
        if (index+1)%20==0:dump(folder/'cache_status.json',{'state':'normal_feature_cache','parents':index+1,'total':len(records),'seconds':time.time()-started})
    assert all(not p.requires_grad and p.grad is None for p in net.parameters());assert sha(V1/'readers'/cat/'best.pt')==digest
    save(path,parents);receipt={'state':'complete','parents':len(records),'views':len(parents),'roles':['train','dev'],'sha256':sha(path),'checkpoint_sha256':digest,'mask_hashes':mask_hashes,'edit_counts':edit_counts,'feature_shape':[48,*SIZE],'frozen_optimizer_parameters':0,'seconds':time.time()-started}
    dump(done,receipt);del net,parents;torch.cuda.empty_cache();return torch.load(path,map_location='cpu',weights_only=False),receipt
