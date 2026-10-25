"""Independent checks using explicitly repeated groups and pairwise AUROC."""
from pathlib import Path
import gzip,json,numpy as np
R=Path(__file__).resolve().parents[1]
with gzip.open(R/'data/predictions.jsonl.gz','rt') as f:rr=[json.loads(s) for s in f]
with np.load(R/'results/group_draws.npz') as z:
    draws=z['multiplicities'];g={s:i for i,s in enumerate(z['group_ids'])}
with np.load(R/'results/bootstrap_metrics.npz') as z:
    boot=z['metrics'];tasks=z['tasks'];methods=z['methods']
checks=0;maxerr=0.
for task in ['screw_bag','Tapes','Tapes_Blurry_CD']:
    ti=next(i for i,t in enumerate(tasks) if t[2]==task)
    for mid,m in enumerate(methods):
        rows=sorted([r for r in rr if r['method']==m and r['task']==task],key=lambda r:r['source_id'])
        for rep in [0,1,17,99,999,9999]:
            weights=np.array([draws[rep,g[r['group_id']]] for r in rows]);ix=np.repeat(np.arange(len(rows)),weights)
            y=np.array([r['anomaly'] for r in rows],bool)[ix];s=np.array([r['score'] for r in rows])[ix];p=np.array([r['predicted_anomaly'] for r in rows],bool)[ix]
            pair=s[y,None]-s[~y][None,:];auc=float(((pair>0)+.5*(pair==0)).mean());tp=(y&p).sum();fp=(~y&p).sum();fn=(y&~p).sum();f1=2*tp/(2*tp+fp+fn)
            error=float(abs(np.array([auc,f1])-boot[rep,ti,mid,:2]).max());assert error<2e-12;maxerr=max(maxerr,error);checks+=1
result={'state':'passed','brute_force_expanded_sample_comparisons':checks,'max_error':maxerr,'implementation':'Explicitly repeat sampled groups, pairwise-positive/negative AUROC, direct confusion counts; does not use the weighted bootstrap metric implementation.'}
print(json.dumps(result,indent=2))
(R/'audit/bootstrap_independent_check.json').write_text(json.dumps(result,indent=2)+'\n')
