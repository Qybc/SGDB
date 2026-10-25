"""Original scoring formula, new normal references, and independently held CAL."""
import argparse,sys,time
from collections import defaultdict
import numpy as np
from io_utils import *
sys.path.insert(0,str(V1))
from features import raw
from program import residual,score,apply_aux
from evaluate import metric,macro

def select():
    verify();files={p.name:sha(p) for p in ROOT.glob('*.py')};files.update({n:sha(ROOT/n) for n in ['protocol.json','training_lock.json']});selections={}
    for cat in read(ROOT/'protocol.json')['categories']:
        selections[cat]={}
        for v in VARIANTS:
            f=ROOT/'readers'/cat/v;r=read(f/'complete.json');h=read(f/'history.json');evaluated=[x for x in h if 'normal_dev' in x]
            chosen=max(evaluated,key=lambda x:x['normal_dev']['objective']);assert chosen['epoch']==r['selected_epoch'];assert sha(f/'best.pt')==r['checkpoint_sha256']
            selections[cat][v]={'epoch':chosen['epoch'],'normal_dev':chosen['normal_dev'],'checkpoint_sha256':r['checkpoint_sha256']}
            for name in ['best.pt','complete.json','history.json','normal_validation.json']:
                p=f/name;files[str(p.relative_to(ROOT))]=sha(p)
    dump(ROOT/'normal_selection.json',{'state':'frozen_normal_DEV_only','chosen':selections,'main_variant':'evidence_edit','test_selection':False,'time_unix':time.time()});files['normal_selection.json']=sha(ROOT/'normal_selection.json')
    dump(ROOT/'selection_lock.json',{'state':'frozen_before_CAL_and_test','files':files,'time_unix':time.time()})

def calibrate():
    lock=read(ROOT/'selection_lock.json')
    for n,h in lock['files'].items():assert sha(ROOT/n)==h
    cats=read(ROOT/'protocol.json')['categories'];original=read(ZL1/'deployment.json')
    for v in VARIANTS:
        deployment={}
        for cat in cats:
            normal=rows(ROOT/'variants'/v/'readers'/cat/'normal_facts.jsonl');train=[r for r in normal if r['split']=='train'];dev=[r for r in normal if r['split']=='dev'];cal=[r for r in normal if r['split']=='calibration'];assert train and dev and cal
            aux=original[cat].get('aux');values=[raw(cat,apply_aux(r['facts'],aux))[1] for r in train];dv=[raw(cat,apply_aux(r['facts'],aux))[1] for r in dev];intervals={}
            for key in sorted(set().union(*(set(a) for a in values))):
                x=[a[key] for a in values if key in a];d=[a[key] for a in dv if key in a]
                if len(x)<3:continue
                lo,hi=np.quantile(x,[.01,.99]);scale=max(float(hi-lo),abs(float(np.median(x)))*.1,.01);pad=max(.25*scale,max((max(lo-a,a-hi,0.) for a in d),default=0.))
                intervals[key]={'lo':float(lo-pad),'hi':float(hi+pad),'scale':scale}
            config={'aux':aux,'intervals':intervals,'references':{},'thresholds':{},'train_ids':[r['source_id'] for r in train],'dev_ids':[r['source_id'] for r in dev],'calibration_ids':[r['source_id'] for r in cal]}
            for row in train:
                rr,_=residual(cat,row['facts'],config);refs=config['references'].setdefault(row['task'],{})
                for key,value in rr.items():refs.setdefault(key,[]).append(value)
            for task in sorted({r['task'] for r in cal}):
                ss=[score(cat,r['facts'],config,task)['score'] for r in cal if r['task']==task]
                config['thresholds'][task]={'threshold':float(np.quantile(ss,.95,method='higher')),'calibration_scores':ss,'n':len(ss),'decision':'strict score > threshold'}
            deployment[cat]=config
        dump(ROOT/'variants'/v/'deployment.json',deployment)
    files={p.name:sha(p) for p in ROOT.glob('*.py')};files.update({n:sha(ROOT/n) for n in ['protocol.json','training_lock.json','selection_lock.json','normal_selection.json']})
    for v in VARIANTS:
        files['variants/'+v+'/deployment.json']=sha(ROOT/'variants'/v/'deployment.json')
        for cat in cats:
            p=ROOT/'variants'/v/'readers'/cat/'normal_facts.jsonl';files[str(p.relative_to(ROOT))]=sha(p)
    dump(ROOT/'execution_lock.json',{'state':'frozen_before_test_inference','created_unix':time.time(),'files':files,'test_truth_sha256':sha(V1/'evaluation_truth.json'),'public_tests_previously_seen':True})

def evaluate():
    lock=read(ROOT/'execution_lock.json')
    for n,h in lock['files'].items():assert sha(ROOT/n)==h,n
    cats=read(ROOT/'protocol.json')['categories'];truth=[r for r in read(V1/'evaluation_truth.json') if r['category'] in cats];idx={(r['dataset'],r['source_id']):r for r in truth};assert len(idx)==9031
    dump(ROOT/'artifacts'/'evaluation_truth.json',truth);manifest=[r for r in read(V1/'manifest.json') if r['category'] in cats];dump(ROOT/'artifacts'/'manifest.json',manifest)
    dump(ROOT/'artifacts'/'training_records.json',[r for r in read(V1/'training_records.json') if r['category'] in cats])
    for cat in cats:
        assert sha(V1/'readers'/cat/'best.pt')==read(ROOT/'training_lock.json')['checkpoint_sha256'][cat]
    summaries=[]
    baseline=[r for r in read(ZL1/'evaluation/predictions.json') if r['category'] in cats]
    for v in VARIANTS+['FMP-v1+ZL1']:
        if v=='FMP-v1+ZL1':pred=baseline
        else:
            deployment=read(ROOT/'variants'/v/'deployment.json');pred=[]
            for cat in cats:
                for row in rows(ROOT/'variants'/v/'readers'/cat/'test_facts.jsonl'):
                    ref=idx[row['dataset'],row['source_id']];assert row['sha256']==ref['sha256'];r=score(cat,row['facts'],deployment[cat],row['task']);threshold=deployment[cat]['thresholds'][row['task']]['threshold'];pred.append({**ref,**r,'threshold':threshold,'predicted_anomaly':r['score']>threshold})
        assert len(pred)==len({(r['dataset'],r['source_id']) for r in pred})==9031
        groups=defaultdict(list)
        for r in pred:groups[r['dataset'],r['category'],r['task']].append(r)
        per=[{'dataset':d,'category':c,'task':t,**metric(rr)} for (d,c,t),rr in sorted(groups.items())]
        summary=[{'method':v,'dataset':d,'scope':scope,'tasks':len(rr),'n':sum(r['n'] for r in rr),**macro(rr)} for d,scope in [('LOCO','logical_only'),('VID-AD','full50')] for rr in [[r for r in per if r['dataset']==d]]]
        assert [(r['tasks'],r['n']) for r in summary]==[(5,1136),(50,7895)]
        out=ROOT/'evaluation'/v
        for name,value in [('predictions.json',pred),('per_task.json',per),('summary.json',summary)]:dump(out/name,value)
        summaries+=summary
    dump(ROOT/'comparison.json',summaries);print(json.dumps(summaries),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','calibrate','evaluate']);a=p.parse_args();globals()[a.mode]()
