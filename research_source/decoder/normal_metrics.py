import numpy as np
from scipy.optimize import linear_sum_assignment
def metrics(objects,truth,classes):
    result=[];ge=[]
    for c in range(classes):
        pp=[o for o in objects if o['class']==c];gg=[o for o in truth if o['class']==c];tp=0
        if pp and gg:
            d=np.linalg.norm(np.array([o['center'] for o in pp])[:,None]-np.array([o['center'] for o in gg])[None],axis=2);ii,jj=linear_sum_assignment(d)
            for i,j in zip(ii,jj):
                if d[i,j]<.08:
                    tp+=1;a=np.array(pp[i]['axis']);z=np.array(gg[j]['axis']);ge.append(float(min(np.abs(a-z).mean(),np.abs(a-z[::-1]).mean())))
        result.append({'class':c,'tp':tp,'fp':len(pp)-tp,'fn':len(gg)-tp,'count_exact':len(pp)==len(gg)})
    return {'per_class':result,'axis_errors':ge}
def aggregate(records,classes):
    per=[]
    for c in range(classes):
        rr=[r['metrics']['per_class'][c] for r in records];tp=sum(r['tp'] for r in rr);fp=sum(r['fp'] for r in rr);fn=sum(r['fn'] for r in rr)
        mask=[r['mask'][c] for r in records if 'mask' in r];inter=sum(a[0] for a in mask);ps=sum(a[1] for a in mask);gs=sum(a[2] for a in mask)
        per.append({'class':c,'tp':tp,'fp':fp,'fn':fn,'point_f1':2*tp/max(1,2*tp+fp+fn),'dice':(2*inter+1)/(ps+gs+1),'count_exact':float(np.mean([r['count_exact'] for r in rr]))})
    f1=float(np.mean([r['point_f1'] for r in per]));dice=float(np.mean([r['dice'] for r in per]));errors=[x for r in records for x in r['metrics']['axis_errors']]
    return {'objective':.8*f1+.2*dice,'point_f1':f1,'mask_dice':dice,'axis_error':float(np.mean(errors)) if errors else None,'count_exact':float(np.mean([r['count_exact'] for r in per])),'per_class':per,'images':len(records)}
