"""One rule executor, normalization, unknown policy and threshold recipe."""
import copy,math,time
import numpy as np
from common import *
from features import raw
from loco_features import classifier_predict
def apply_aux(f,aux):
 if aux:
  f=copy.deepcopy(f);kind,prob=classifier_predict(f.get('liquid_colour_descriptor'),aux);f.update(liquid_type=kind,liquid_type_probabilities=prob)
 return f
def fit_color(rows):
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 labels={r['source_id']:r for r in read(ROOT/'training_records.json') if r['category']=='juice_bottle' and r['split']=='train'};xy=[]
 for r in rows:
  if r['split']!='train' or r['source_id'] not in labels:continue
  x=r['facts'].get('liquid_colour_descriptor');rr=labels[r['source_id']];kind=rr.get('stratum')
  if kind not in ['orange','banana','cherry']:
   kind=next((CATEGORIES['juice_bottle'][o['class']] for o in rr['objects'] if o['class']>=4),None)
  if x is not None and kind in ['orange','banana','cherry']:xy.append((x,kind))
 assert len(xy)>=10;x=np.array([x for x,y in xy]);y=[y for x,y in xy];sc=StandardScaler().fit(x);cl=LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(sc.transform(x),y)
 return {'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'coef':cl.coef_.tolist(),'intercept':cl.intercept_.tolist(),'classes':cl.classes_.tolist(),'train_examples':len(xy)}
def residual(cat,f,config):
 rr,mm,unknown=raw(cat,apply_aux(f,config.get('aux')))
 for k,s in config['intervals'].items():
  if k not in mm:
   # A conditional measurement is required only when its category condition applies.
   if k.startswith('cable_length_') and not k.endswith(str(f.get('cable_color'))):continue
   unknown.append(k);continue
  rr[k]=float(max(0.,s['lo']-mm[k],mm[k]-s['hi'])/s['scale'])
 # Explicit unavailable evidence is retained and contributes exactly one residual.
 rr['measurement_unavailable']=float(bool(unknown));return rr,sorted(set(unknown))
def score(cat,f,config,task):
 rr,unknown=residual(cat,f,config);norm={};refs=config['references'][task]
 for k,v in rr.items():
  ref=refs.get(k,[0.]);tail=(1+sum(x>=v for x in ref))/(len(ref)+1);scale=max(float(np.quantile(ref,.75)-np.quantile(ref,.25)),1.);norm[k]=float(-math.log(tail)+max(0.,v-max(ref))/scale)
 return {'score':float(max(norm.values(),default=0.)+.01*sum(norm.values())),'rule_residuals':rr,'normalized_residuals':norm,'unknown':unknown,'evidence_status':'insufficient' if unknown else 'available','rule_violations':[k for k,v in rr.items() if v>1e-8]}
def fit_all():
 assert not (ROOT/'execution_lock.json').exists();deploy={}
 for cat in CATEGORIES:
  rows=lines(ROOT/'readers'/cat/'normal_facts.jsonl');tr=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];cal=[r for r in rows if r['split']=='calibration'];assert tr and dev and cal
  aux=fit_color(rows) if cat=='juice_bottle' else None;values=[raw(cat,apply_aux(r['facts'],aux))[1] for r in tr];devvalues=[raw(cat,apply_aux(r['facts'],aux))[1] for r in dev];intervals={}
  for key in sorted(set().union(*[set(x) for x in values])):
   v=[x[key] for x in values if key in x];dv=[x[key] for x in devvalues if key in x]
   if len(v)<3:continue
   lo,hi=np.quantile(v,[.01,.99]);scale=max(float(hi-lo),abs(float(np.median(v)))*.1,.01);pad=max(.25*scale,max([max(lo-x,x-hi,0.) for x in dv],default=0.));intervals[key]={'lo':float(lo-pad),'hi':float(hi+pad),'scale':scale}
  config={'aux':aux,'intervals':intervals,'references':{},'thresholds':{},'train_ids':[r['source_id'] for r in tr],'dev_ids':[r['source_id'] for r in dev],'calibration_ids':[r['source_id'] for r in cal]}
  for r in tr:
   rr,_=residual(cat,r['facts'],config);refs=config['references'].setdefault(r['task'],{})
   for k,v in rr.items():refs.setdefault(k,[]).append(v)
  assert set(r['task'] for r in cal)<=set(config['references'])
  for task in sorted(set(r['task'] for r in cal)):
   scores=[score(cat,r['facts'],config,task)['score'] for r in cal if r['task']==task];config['thresholds'][task]={'threshold':float(np.quantile(scores,.95,method='higher')),'calibration_scores':scores,'n':len(scores),'decision':'strict score > threshold'}
  deploy[cat]=config
 dump(ROOT/'deployment.json',deploy);files={p.name:sha(p) for p in ROOT.glob('*.py')};files.update({n:sha(ROOT/n) for n in ['training_lock.json','protocol.json','manifest.json','evaluation_truth.json','deployment.json','rope_ocr_normal.jsonl','rope_ocr_protocol_lock.json']})
 for cat in CATEGORIES:
  for name in ['best.pt','complete.json','normal_facts.jsonl','normal_facts_complete.json']:
   p=ROOT/'readers'/cat/name;files[str(p.relative_to(ROOT))]=sha(p)
 dump(ROOT/'execution_lock.json',{'state':'frozen_before_test_inference','files':files,'created_unix':time.time(),'test_seen_in_prior_development':True});print('All model/rule/calibration deployments frozen',flush=True)
if __name__=='__main__':fit_all()
