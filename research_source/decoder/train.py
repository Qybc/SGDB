import argparse,fcntl,math,time
import numpy as np,torch
from torch import nn
from io_utils import *
from decoder import *
from cache_data import cache
from normal_metrics import metrics,aggregate

@torch.inference_mode()
def assess(head,parents,keep=None):
    head.eval();records=[];classes=head.classes
    if keep:keep.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(parents),4):
        pp=parents[start:start+4];out=head(pp)
        for b,p in enumerate(pp):
            objects,slots=decoded(out,b);prob=out['masks'][b,slots].sigmoid().cpu().numpy()>.5
            pred=np.zeros((classes,*SIZE),bool);gt=np.zeros_like(pred)
            for o,m in zip(objects,prob):pred[o['class']]|=m
            for o,m in zip(p['truth'],p['masks'].numpy()):gt[o['class']]|=m>0
            row={k:p[k] for k in ['source_id','sha256','split','dataset','category','view','edit','native_size']}
            row.update(objects=objects,truth=p['truth'],metrics=metrics(objects,p['truth'],classes),mask=[[int((a&z).sum()),int(a.sum()),int(z.sum())] for a,z in zip(pred,gt)])
            if keep:
                key=hashlib.sha256((p['source_id']+p['view']).encode()).hexdigest();f=keep/(key+'.npz')
                np.savez_compressed(f,pred=np.packbits(pred.flatten()),truth=np.packbits(gt.flatten()),shape=np.array(pred.shape))
                row.update(mask_file=str(f.relative_to(ROOT)),mask_sha256=sha(f))
            records.append(row)
    return aggregate(records,classes),records

def run(cat):
    folder=ROOT/'readers'/cat;folder.mkdir(parents=True,exist_ok=True);mutex=(folder/'train.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();torch.set_num_threads(2);parents,receipt=cache(cat);train=[parents[i:i+2] for i in range(0,len(parents),2) if parents[i]['split']=='train'];dev=[p for p in parents if p['split']=='dev'];assert train and dev
    baseline=[{'metrics':metrics(p['anchor'],p['truth'],len(CATEGORIES[cat]))} for p in dev]
    dump(folder/'anchor_normal_dev.json',aggregate(baseline,len(CATEGORIES[cat])))
    for variant in VARIANTS:
        path=folder/variant;path.mkdir(exist_ok=True)
        if (path/'complete.json').exists():continue
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
        head=FactDecoder(len(CATEGORIES[cat]),variant).cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
        rng=np.random.default_rng(42);step=0;history=[];best=-1.;values=[];last=path/'last.pt';started=time.time()
        if last.exists():
            ck=torch.load(last,map_location='cpu',weights_only=False);assert ck['training_lock_sha256']==sha(ROOT/'training_lock.json');head.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];history=ck['history'];best=ck['best'];values=ck['values'];rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        def persist():save(last,{'model':head.state_dict(),'optimizer':opt.state_dict(),'step':step,'history':history,'best':best,'values':values,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'training_lock_sha256':sha(ROOT/'training_lock.json')})
        def finish_epoch():
            nonlocal values,best
            epoch=step//24;r={'epoch':epoch,'optimizer_steps':step,'loss':float(np.mean(values)) if values else None,'seconds':time.time()-started}
            if epoch==1 or epoch%8==0:
                m,_=assess(head,dev);r['normal_dev']=m
                if m['objective']>best:
                    best=m['objective'];save(path/'best.pt',{'model':{k:v.detach().cpu() for k,v in head.state_dict().items()},'epoch':epoch,'normal_dev':m,'training_lock_sha256':sha(ROOT/'training_lock.json'),'variant':variant})
            history.append(r);values=[];dump(path/'history.json',history);dump(path/'status.json',{'state':'training','category':cat,'variant':variant,**r});persist();print(cat,variant,json.dumps(r),flush=True)
        if step and step%24==0 and len(history)<step//24:finish_epoch()
        while step<32*24:
            opt.param_groups[0]['lr']=3e-4*.5*(1+math.cos(math.pi*(step//24)/32))
            pair=train[int(rng.integers(len(train)))];head.train();opt.zero_grad(set_to_none=True)
            out=head(pair,duplicate=(variant=='evidence_edit' and step%2==0));total,parts=loss(out,pair,variant);assert torch.isfinite(total),(cat,variant,step,parts)
            total.backward();nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();step+=1;values.append(float(total.detach()))
            if step%4==0:persist()
            if step%24==0:finish_epoch()
        ck=torch.load(path/'best.pt',map_location='cpu',weights_only=True);head.load_state_dict(ck['model']);m,rr=assess(head,dev,path/'validation_masks');dump(path/'normal_validation.json',rr)
        assert abs(m['objective']-ck['normal_dev']['objective'])<1e-7
        dump(path/'complete.json',{'state':'complete','category':cat,'variant':variant,'cycles':32,'optimizer_steps':step,'image_exposures':step*2,'selected_epoch':ck['epoch'],'normal_dev':m,'checkpoint_sha256':sha(path/'best.pt'),'training_lock_sha256':sha(ROOT/'training_lock.json'),'frozen_FMP_checkpoint_sha256':receipt['checkpoint_sha256'],'frozen_FMP_optimizer_parameters':0,'parameter_count':sum(p.numel() for p in head.parameters()),'train_parent_ids':[p[0]['source_id'] for p in train],'dev_parent_ids':sorted(set(p['source_id'] for p in dev)),'normal_validation_sha256':sha(path/'normal_validation.json'),'seconds':time.time()-started})
        del head,opt;torch.cuda.empty_cache()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--categories',nargs='+',required=True);a=p.parse_args()
    for c in a.categories:run(c)
