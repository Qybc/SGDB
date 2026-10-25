import argparse,fcntl,sys,time
import numpy as np,torch
from PIL import Image
from io_utils import *
from decoder import *
from cache_data import network,extract
sys.path.insert(0,str(V1))
from features import extract as fact_extract

@torch.inference_mode()
def infer(cat,test):
    lockname='execution_lock.json' if test else 'selection_lock.json';lock=read(ROOT/lockname)
    for p,h in lock['files'].items():assert sha(ROOT/p)==h,(lockname,p)
    prefix='test' if test else 'normal';folder=ROOT/'inference'/cat;folder.mkdir(parents=True,exist_ok=True);mutex=(folder/(prefix+'.lock')).open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    done=folder/(prefix+'_complete.json')
    if done.exists():
        for v,h in read(done)['output_sha256'].items():assert sha(ROOT/'variants'/v/'readers'/cat/(prefix+'_facts.jsonl'))==h
        return
    net,settings,digest=network(cat);heads={};streams={};paths={};temp={}
    for v in VARIANTS:
        ck=torch.load(ROOT/'readers'/cat/v/'best.pt',map_location='cpu',weights_only=True);head=FactDecoder(len(CATEGORIES[cat]),v).cuda().eval();head.load_state_dict(ck['model']);heads[v]=head
        p=ROOT/'variants'/v/'readers'/cat/(prefix+'_facts.jsonl');p.parent.mkdir(parents=True,exist_ok=True);paths[v]=p;temp[v]=Path(str(p)+'.tmp');streams[v]=temp[v].open('w')
    source=V1/'readers'/cat/(prefix+'_facts.jsonl');expected=read(source.parent/(prefix+'_facts_complete.json'));assert sha(source)==expected['output_sha256']
    count=0;started=time.time();evidence_hashes={};example_masks=0
    for r in rows(source):
        assert r['split'] in (['test'] if test else ['train','dev','calibration']);assert sha(r['image_path'])==r['sha256']
        im=Image.open(r['image_path']).convert('RGB');feat,_,seg=extract(net,np.array(im),settings)
        frozen_masks=(seg.numpy()>np.array([s['seg_threshold'] for s in settings])[:,None,None]).astype('float32')
        parent={'features':feat,'anchor':r['objects'],'native_size':list(im.size)}
        for v,head in heads.items():
            out=head([parent]);objects,slots=decoded(out,0)
            ocr=r['facts'].get('attributes',{}).get('label_text');facts=fact_extract(cat,im,objects,frozen_masks,ocr)
            # Keep frozen visual/OCR implementations; only decoded object facts and their consequences change.
            support=out['masks'][0,slots].sigmoid().cpu().numpy()>.5
            if test and count<3:
                p=ROOT/'examples'/v/cat/(str(count)+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,support=np.packbits(support.flatten()),shape=np.array(support.shape));evidence_hashes[str(p.relative_to(ROOT))]=sha(p);example_masks+=1
            result={**r,'objects':objects,'facts':facts,'deep_decoder':{'variant':v,'checkpoint_sha256':sha(ROOT/'readers'/cat/v/'best.pt'),'slot_indices':slots,'support_areas':support.sum((1,2)).astype(int).tolist(),'native_size':list(im.size)}}
            streams[v].write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
        count+=1
        if count%25==0:dump(folder/(prefix+'_status.json'),{'state':'inference','images':count,'seconds':time.time()-started})
    for v in VARIANTS:streams[v].close();temp[v].replace(paths[v])
    assert sha(V1/'readers'/cat/'best.pt')==digest;assert all(not p.requires_grad and p.grad is None for p in net.parameters())
    dump(done,{'state':'complete','category':cat,'images':count,'frozen_checkpoint_sha256':digest,'source_sha256':expected['output_sha256'],'output_sha256':{v:sha(p) for v,p in paths.items()},'support_examples':example_masks,'evidence_hashes':evidence_hashes,'new_optimizer_steps':0,'seconds':time.time()-started});print(cat,prefix,count,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--categories',nargs='+',required=True);p.add_argument('--test',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    for c in a.categories:infer(c,a.test)
