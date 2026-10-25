"""CPU-only reproduction of frozen metrics, paired grouped intervals and transitions.

No torch, image loader, training, inference or calibration code is imported.
"""
from pathlib import Path
import argparse, collections, csv, gzip, hashlib, json, platform, sys, time
import numpy as np

def load(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rr):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
def metric(y,s,p,w=None):
    w=np.ones(len(y)) if w is None else w
    pos=w[y].sum();neg=w[~y].sum()
    order=np.argsort(s,kind='stable');ss=s[order]
    starts=np.r_[0,np.where(ss[1:]!=ss[:-1])[0]+1]
    wp=np.add.reduceat((w*y)[order],starts);wn=np.add.reduceat((w*~y)[order],starts)
    auc=float((wp*(np.cumsum(wn)-.5*wn)).sum()/(pos*neg))
    tp=float(w[y&p].sum());fp=float(w[~y&p].sum());fn=float(w[y&~p].sum());tn=float(w[~y&~p].sum())
    return {'AUROC':auc,'F1':2*tp/max(1,2*tp+fp+fn),'normal_FPR':fp/max(1,fp+tn),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),'n':len(y)}

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a.add_argument('--out',type=Path,default=None);args=a.parse_args()
    root=args.root.resolve();out=(args.out or root/'results').resolve()
    assert not out.exists(),'Use a new --out directory; archived results are immutable.'
    out.mkdir(parents=True);started=time.time()
    protocol=json.loads((root/'analysis_protocol.json').read_text());B=protocol['repetitions'];seed=protocol['seed']
    allrows=load(root/'data/predictions.jsonl.gz');by=collections.defaultdict(dict)
    for r in allrows:
        assert r['source_id'] not in by[r['method']]
        assert bool(r['score']>r['threshold'])==r['predicted_anomaly']
        by[r['method']][r['source_id']]=r
    methods=sorted(by);ids=sorted(by['SGDB_V2']);assert len(ids)==9031
    for m in methods:assert set(by[m])==set(ids)
    meta=[by['SGDB_V2'][sid] for sid in ids];y=np.array([r['anomaly'] for r in meta],bool)
    categories=sorted({(r['dataset'],r['category']) for r in meta});tasks=sorted({(r['dataset'],r['category'],r['task']) for r in meta})
    metrics=[]
    for m in methods:
        for d,c,t in tasks:
            ix=[i for i,r in enumerate(meta) if r['task']==t and r['dataset']==d]
            rr=[by[m][ids[i]] for i in ix];yy=y[ix];ss=np.array([r['score'] for r in rr]);pp=np.array([r['predicted_anomaly'] for r in rr],bool)
            z=metric(yy,ss,pp)
            # An independently expressed rank-sum AUC protects the weighted implementation.
            npos=int(yy.sum());nneg=len(yy)-npos
            comparisons=ss[yy,None]-ss[~yy][None,:]
            check=float(((comparisons>0)+.5*(comparisons==0)).mean())
            assert abs(z['AUROC']-check)<1e-12
            metrics.append({'method':m,'dataset':d,'category':c,'task':t,**z})
    csvout(out/'per_task.csv',metrics)
    summaries=[]
    for m in methods:
        for d in ['LOCO','VID-AD']:
            rr=[r for r in metrics if r['method']==m and r['dataset']==d]
            summaries.append({'method':m,'dataset':d,'tasks':len(rr),'n':sum(r['n'] for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['AUROC','F1','normal_FPR']},'errors':sum(r['FP']+r['FN'] for r in rr)})
    csvout(out/'metrics.csv',summaries);dump(out/'metrics.json',summaries)
    # Validate every available original manuscript task row, retaining historical IDs.
    mapping={'hybrid_evidence_edit':'SGDB_V2','hybrid_uncertified':'initial_bound','plain_set':'direct_plain_set','evidence_edit':'direct_evidence_edit'}
    reference=list(csv.DictReader((root/'data/manuscript_task_reference.csv').open()))
    lookup={(r['method'],r['dataset'],r['task']):r for r in metrics};checked=0
    for r in reference:
        m=mapping.get(r['method'],r['method']);key=(m,r['dataset'],r['task'])
        if key not in lookup:raise AssertionError(('Missing original task row',key))
        actual=lookup[key]
        for k in ['AUROC','F1','normal_FPR']:
            assert abs(actual[k]-float(r[k]))<1e-12,(key,k,actual[k],r[k])
        for k in ['TP','FP','FN','TN','n']:assert actual[k]==int(r[k]),(key,k)
        checked+=1
    # The same original group has one sampled multiplicity for every method and view.
    gids=sorted({r['group_id'] for r in meta});gi={g:i for i,g in enumerate(gids)};rowgroups=np.array([gi[r['group_id']] for r in meta])
    gm=collections.defaultdict(list)
    for r in meta:gm[r['group_id']].append(r)
    strata=collections.defaultdict(list)
    for g in gids:
        rr=gm[g];assert len({(r['category'],r['dataset'],r['anomaly']) for r in rr})==1
        r=rr[0];strata[r['dataset'],r['category'],r['anomaly']].append(gi[g])
    rng=np.random.default_rng(seed);draws=np.zeros((B,len(gids)),np.uint16)
    for key,index in sorted(strata.items()):
        n=len(index);draws[:,index]=rng.multinomial(n,np.full(n,1/n),size=B).astype(np.uint16)
    np.savez_compressed(out/'group_draws.npz',multiplicities=draws,group_ids=np.array(gids),seed=np.array(seed))
    chosen=['initial_bound','uniform_mask','quality_only','SGDB_V2','uniform_mask_fixed_refs','quality_only_fixed_refs','SGDB_V2_fixed_refs','FMP-v1+ZL1']
    boot=np.empty((B,len(tasks),len(chosen),3),float)
    for ti,(d,c,t) in enumerate(tasks):
        ix=np.array([i for i,r in enumerate(meta) if r['task']==t and r['dataset']==d]);yy=y[ix]
        for mid,m in enumerate(chosen):
            ss=np.array([by[m][ids[i]]['score'] for i in ix]);pp=np.array([by[m][ids[i]]['predicted_anomaly'] for i in ix],bool)
            order=np.argsort(ss,kind='stable');sorted_s=ss[order];starts=np.r_[0,np.flatnonzero(sorted_s[1:]!=sorted_s[:-1])+1]
            for first in range(0,B,250):
                last=min(B,first+250);w=draws[first:last,rowgroups[ix]].astype(float)
                bp=np.add.reduceat((w*yy)[:,order],starts,axis=1);bn=np.add.reduceat((w*~yy)[:,order],starts,axis=1)
                auc=(bp*(np.cumsum(bn,axis=1)-.5*bn)).sum(1)/(bp.sum(1)*bn.sum(1))
                tp=w[:,yy&pp].sum(1);fp=w[:,~yy&pp].sum(1);fn=w[:,yy&~pp].sum(1);tn=w[:,~yy&~pp].sum(1)
                boot[first:last,ti,mid,0]=auc
                boot[first:last,ti,mid,1]=2*tp/np.maximum(1,2*tp+fp+fn)
                boot[first:last,ti,mid,2]=fp/(fp+tn)
        if ti%10==0:print('Grouped bootstrap tasks',ti+1,'/',len(tasks),flush=True)
    assert np.isfinite(boot).all()
    np.savez_compressed(out/'bootstrap_metrics.npz',metrics=boot,tasks=np.array(tasks),methods=np.array(chosen),metrics_names=np.array(['AUROC','F1','normal_FPR']))
    intervals=[];paired=[]
    scopes=[(d,'macro',[i for i,t in enumerate(tasks) if t[0]==d]) for d in ['LOCO','VID-AD']]
    scopes += [(d,c,[i for i,t in enumerate(tasks) if t[0]==d and t[1]==c]) for d,c in categories]
    for d,scope,indices in scopes:
        b=boot[:,indices].mean(1)
        for j,m in enumerate(chosen):
            point=[lookup[m,d,tasks[i][2]] for i in indices]
            for k,key in enumerate(['AUROC','F1','normal_FPR']):
                low,high=np.quantile(b[:,j,k],[.025,.975],method='linear')
                intervals.append({'dataset':d,'scope':scope,'method':m,'metric':key,'estimate_percent':100*float(np.mean([r[key] for r in point])),'lower95_percent':100*float(low),'upper95_percent':100*float(high),'replicates':B})
        for target,control,kind in [('SGDB_V2','initial_bound','primary'),('SGDB_V2','uniform_mask','primary'),('SGDB_V2','quality_only','primary'),('SGDB_V2','FMP-v1+ZL1','secondary'),('SGDB_V2_fixed_refs','initial_bound','fixed_refs'),('SGDB_V2_fixed_refs','uniform_mask_fixed_refs','fixed_refs'),('SGDB_V2_fixed_refs','quality_only_fixed_refs','fixed_refs')]:
            j=chosen.index(target);l=chosen.index(control)
            for k,key in enumerate(['AUROC','F1']):
                delta=b[:,j,k]-b[:,l,k];low,high=np.quantile(delta,[.025,.975],method='linear')
                point=np.mean([lookup[target,d,tasks[i][2]][key]-lookup[control,d,tasks[i][2]][key] for i in indices])
                paired.append({'dataset':d,'scope':scope,'target':target,'control':control,'kind':kind,'metric':key,'difference_pp':100*float(point),'lower95_pp':100*float(low),'upper95_pp':100*float(high),'replicates':B})
    csvout(out/'intervals.csv',intervals);csvout(out/'paired_intervals.csv',paired);dump(out/'paired_intervals.json',paired)
    transitions=[];grouptrans=[];count=[]
    for control in ['initial_bound','uniform_mask','quality_only']:
        cnt=collections.Counter();gg=collections.defaultdict(collections.Counter)
        for sid in ids:
            a=by[control][sid];b=by['SGDB_V2'][sid];old=a['anomaly']==a['predicted_anomaly'];new=b['anomaly']==b['predicted_anomaly']
            kind=('repaired' if new else 'regressed') if old!=new else ('correct_unchanged' if new else 'wrong_unchanged')
            cnt[kind]+=1;gg[b['group_id']][kind]+=1
            if old!=new:
                outcome=('FN_to_TP' if b['anomaly'] else 'FP_to_TN') if new else ('TP_to_FN' if b['anomaly'] else 'TN_to_FP')
                transitions.append({'control':control,'source_id':sid,'group_id':b['group_id'],'category':b['category'],'capture_condition':b['capture_condition'],'transition':outcome,'old_score':a['score'],'old_threshold':a['threshold'],'new_score':b['score'],'new_threshold':b['threshold']})
        for g,cc in gg.items():
            rr=gm[g];state='both' if cc['repaired'] and cc['regressed'] else 'repair_only' if cc['repaired'] else 'regress_only' if cc['regressed'] else 'unchanged'
            grouptrans.append({'control':control,'group_id':g,'dataset':rr[0]['dataset'],'category':rr[0]['category'],'state':state,'repaired_views':cc['repaired'],'regressed_views':cc['regressed'],'unchanged_views':cc['correct_unchanged']+cc['wrong_unchanged'],'net_fewer_errors':cc['repaired']-cc['regressed']})
        count.append({'control':control,**dict(cnt),'repair_groups':sum(bool(v['repaired']) for v in gg.values()),'regress_groups':sum(bool(v['regressed']) for v in gg.values()),'groups_with_both':sum(bool(v['repaired'] and v['regressed']) for v in gg.values())})
    csvout(out/'view_transitions.csv',transitions);csvout(out/'group_transitions.csv',grouptrans);dump(out/'transition_summary.json',count)
    categories_out=[];conditions=[]
    for m in chosen:
        for d,c in categories:
            rr=[r for r in metrics if r['method']==m and r['dataset']==d and r['category']==c]
            categories_out.append({'method':m,'dataset':d,'category':c,'tasks':len(rr),'n':sum(r['n'] for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['AUROC','F1','normal_FPR']}})
        for cond in sorted({r['capture_condition'] for r in meta if r['dataset']=='VID-AD'}):
            tt={r['task'] for r in meta if r['dataset']=='VID-AD' and r['capture_condition']==cond}
            rr=[r for r in metrics if r['method']==m and r['dataset']=='VID-AD' and r['task'] in tt]
            conditions.append({'method':m,'capture_condition':cond,'tasks':len(rr),'n':sum(r['n'] for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['AUROC','F1','normal_FPR']}})
    csvout(out/'per_category.csv',categories_out);csvout(out/'per_condition.csv',conditions)
    assert not any(k in sys.modules for k in ['torch','transformers','tensorflow'])
    dump(out/'run_receipt.json',{'state':'passed','original_task_rows_verified':checked,'methods':len(methods),'predictions_verified':len(allrows),'bootstrap_replicates':B,'group_count':len(gids),'training':0,'model_forward_calls':0,'baseline_runs':0,'threshold_refits':0,'python':platform.python_version(),'numpy':np.__version__,'seconds':round(time.time()-started,2)})
    print(json.dumps(count,indent=2));print('PASS',out,flush=True)

if __name__=='__main__':main()
