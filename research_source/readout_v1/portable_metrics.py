import numpy as np
from scipy.stats import rankdata

def metric(rr):
    yy=np.array([r['anomaly'] for r in rr],bool);pp=np.array([r['predicted_anomaly'] for r in rr],bool);ss=np.array([r['score'] for r in rr]);tp=int((yy&pp).sum());fp=int((~yy&pp).sum());fn=int((yy&~pp).sum());tn=int((~yy&~pp).sum());a=int(yy.sum());b=len(yy)-a
    return {'AUROC':float((rankdata(ss)[yy].sum()-a*(a+1)/2)/(a*b)),'F1':2*tp/max(1,2*tp+fp+fn),'recall':tp/max(1,tp+fn),'normal_FPR':fp/max(1,fp+tn),'unknown_rate':sum(bool(r['unknown']) for r in rr)/len(rr),'n':len(rr),'TP':tp,'FP':fp,'FN':fn,'TN':tn}
