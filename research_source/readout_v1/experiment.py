"""CPU replay of frozen masks and SGDB facts. No neural libraries or model calls."""
from pathlib import Path
import collections,copy,csv,gzip,hashlib,json,math,sys,time,os
import numpy as np
from scipy import ndimage
from PIL import Image
W=Path(__file__).resolve().parent
BASE=Path(os.environ.get('SGDB_ARCHIVE_ROOT', str(W.parents[1]/'archival_inputs')))
HY=BASE/'outputs/混合事实解码_HDZ_20261021'
ZL=BASE/'outputs/轴长读取改进_ZL1_20261019/evidence/readers/screw_bag'
CODE=W.parent/'binding'
sys.path.insert(0,str(CODE))
from hybrid import facts
from original_executor import LABELS,raw,residual,score,apply_aux,relation
from portable_metrics import metric
OUT=W/'results'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:return [json.loads(t) for t in f]
def save(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(p,'wt') as f:
  for r in x:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def mask_index(role):
 ev=read(ZL/(role+'_axis_evidence.json'));zz=np.load(ZL/(role+'_masks.npz'))['masks'];assert len(ev)==len(zz)
 return {r['source_id']:(r,z) for r,z in zip(ev,zz)}
def get_mask(cat,r,zi,mi):
 if cat=='screw_bag':
  e,z=zi[r['source_id']];assert e['sha256']==r['sha256'];h,w=e['mask_shape'];return {2:np.unpackbits(z)[:h*w].reshape(h,w).astype(bool)}
 e=mi[r['source_id']];assert e['sha256']==r['sha256'];p=W/'inputs'/e['archive_path'];assert sha(p)==e['mask_sha256'];mask=np.array(Image.open(p));return {k:mask==k+1 for k in PROTOCOL['target_classes'][cat]}
def axis_quality(axis,native,instance_mask,size):
 axis=np.array(axis)*size;delta=axis[1]-axis[0];length=float(np.linalg.norm(delta))
 if length<1e-8:return {'quality':0.,'line_support':0.,'extent_coverage':0.}
 direction=delta/length;h,w=instance_mask.shape
 dilated=ndimage.binary_dilation(instance_mask,structure=np.ones((3,3)),iterations=1)
 line=axis[0]+np.linspace(0,1,64)[:,None]*delta
 ij=np.floor(line/size*np.array([w,h])).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
 member=np.zeros(64,bool);member[inside]=dilated[ij[inside,1],ij[inside,0]];precision=float(member.mean())
 projection=(native-axis[0])@direction;pad=float(np.linalg.norm(size/np.array([w,h])));coverage=float(((projection>=-pad)&(projection<=length+pad)).mean())
 labels,n=ndimage.label(instance_mask,structure=np.ones((3,3)));counts=np.bincount(labels.ravel())[1:];purity=float(counts.max()/len(native)) if len(counts) else 0.
 return {'quality':float(math.sqrt(precision*coverage)*purity),'line_support':precision,'extent_coverage':coverage,'component_purity':purity,'components':int(n),'length_px':length}
def candidate_record(cat,r,masks):
 size=np.array(r['native_size'],float);events=[]
 for cls,mask in masks.items():
  indices=[i for i,o in enumerate(r['objects']) if o['class']==cls and o['center'] is not None and o['axis'] is not None]
  yy,xx=np.where(mask);h,w=mask.shape
  points=np.column_stack([(xx+.5)/w,(yy+.5)/h]);native=points*size
  if not indices:continue
  centers=np.array([r['objects'][i]['center'] for i in indices])*size
  owner=np.argmin(((native[:,None,:]-centers[None,:,:])**2).sum(2),axis=1) if len(points) else np.empty(0,int)
  for j,i in enumerate(indices):
   p=native[owner==j];binary=np.zeros(mask.shape,bool);binary[yy[owner==j],xx[owner==j]]=True
   event={'object_index':i,'class':cls,'label':LABELS[cat][cls],'assigned_pixels':len(p),'old_axis':r['objects'][i]['axis'],'mask_axis':None,'readable':False}
   if len(p)>=PROTOCOL['mask_readout']['minimum_assigned_pixels']:
    center=p.mean(0);values,vectors=np.linalg.eigh(np.cov((p-center).T));direction=vectors[:,-1];projection=(p-center)@direction
    lo,hi=np.quantile(projection,[.01,.99]);axis=np.stack([center+lo*direction,center+hi*direction])/size
    if np.isfinite(axis).all() and hi-lo>1e-8:
     event.update(readable=True,mask_axis=axis.tolist(),old=axis_quality(r['objects'][i]['axis'],p,binary,size),mask=axis_quality(axis,p,binary,size))
   events.append(event)
 return events
def transform(cat,r,events,setting,floor):
 objects=copy.deepcopy(r['objects']);changed=[]
 for e in events:
  use=e['readable'] and (setting=='uniform_mask' or (e['mask']['quality']>=floor and e['mask']['quality']>=e['old']['quality']+.05))
  if use:objects[e['object_index']]['axis']=e['mask_axis'];changed.append(e['object_index'])
 ff=facts(cat,r['facts'],objects,LABELS[cat],np.array(r['native_size']),relation)
 assert ff['counts']==r['facts']['counts']
 for old,new in zip(r['objects'],objects):assert {k:v for k,v in old.items() if k!='axis'}=={k:v for k,v in new.items() if k!='axis'}
 return {**{k:r[k] for k in ['source_id','sha256','split','task','category','dataset','native_size','association']},'objects':objects,'facts':ff,'changed_axes':changed}
def calibrate(cat,rr,aux):
 train=[r for r in rr if r['split']=='train'];dev=[r for r in rr if r['split']=='dev'];cal=[r for r in rr if r['split']=='calibration'];assert train and dev and cal
 tv=[raw(cat,apply_aux(r['facts'],aux))[1] for r in train];dv=[raw(cat,apply_aux(r['facts'],aux))[1] for r in dev];intervals={}
 for key in sorted(set().union(*(set(x) for x in tv))):
  x=[r[key] for r in tv if key in r];d=[r[key] for r in dv if key in r]
  if len(x)<3:continue
  lo,hi=np.quantile(x,[.01,.99]);scale=max(float(hi-lo),abs(float(np.median(x)))*.1,.01);pad=max(.25*scale,max((max(lo-y,y-hi,0.) for y in d),default=0.));intervals[key]={'lo':float(lo-pad),'hi':float(hi+pad),'scale':scale}
 cfg={'aux':aux,'intervals':intervals,'references':{},'thresholds':{},'train_ids':[r['source_id'] for r in train],'dev_ids':[r['source_id'] for r in dev],'calibration_ids':[r['source_id'] for r in cal]}
 for r in train:
  res,_=residual(cat,r['facts'],cfg);ref=cfg['references'].setdefault(r['task'],{})
  for k,v in res.items():ref.setdefault(k,[]).append(v)
 for task in sorted({r['task'] for r in cal}):
  ss=[score(cat,r['facts'],cfg,task)['score'] for r in cal if r['task']==task];cfg['thresholds'][task]={'threshold':float(np.quantile(ss,.95,method='higher')),'calibration_scores':ss,'n':len(ss),'decision':'strict score > threshold'}
 return cfg
def classify(cat,r,cfg,original):
 q=score(cat,r['facts'],cfg,r['task']);t=cfg['thresholds'][r['task']]['threshold'];return {**original,**q,'threshold':t,'predicted_anomaly':q['score']>t}
def normalize_parent(sid):
 parts=sid.split('/');parts[0]=parts[0].split('_Cable_BG')[0].split('_Mesh_BG')[0].split('_Low-light_CD')[0].split('_Blurry_CD')[0];return '/'.join(parts)
def summarize(predictions,original):
 per=[]
 groups=collections.defaultdict(list)
 for r in predictions:groups[r['dataset'],r['category'],r['task']].append(r)
 for (d,c,t),rr in sorted(groups.items()):per.append({'dataset':d,'category':c,'task':t,**metric(rr)})
 summaries=[]
 for dataset in ['LOCO','VID-AD']:
  for scope in ['full','affected']:
   pp=[r for r in per if r['dataset']==dataset and (scope=='full' or r['category'] in PROTOCOL['categories'])];rr=[r for r in predictions if r['dataset']==dataset and (scope=='full' or r['category'] in PROTOCOL['categories'])]
   if not pp:continue
   counter=collections.Counter();repairs=[];regressions=[]
   for q in rr:
    o=original[q['source_id']];oc=o['anomaly']==o['predicted_anomaly'];nc=q['anomaly']==q['predicted_anomaly']
    counter['original_correct' if oc else 'original_wrong']+=1
    if not oc and nc:counter['repaired']+=1;counter['FN_to_TP' if o['anomaly'] else 'FP_to_TN']+=1;repairs.append(q['source_id'])
    if oc and not nc:counter['regressed']+=1;counter['TP_to_FN' if o['anomaly'] else 'TN_to_FP']+=1;regressions.append(q['source_id'])
   summaries.append({'dataset':dataset,'scope':scope,'tasks':len(pp),'n':len(rr),**{k:float(np.mean([r[k] for r in pp])) for k in ['AUROC','F1','recall','normal_FPR','unknown_rate']},**dict(counter),'net_repaired':counter['repaired']-counter['regressed'],'repair_parent_count':len({normalize_parent(s) for s in repairs}),'regress_parent_count':len({normalize_parent(s) for s in regressions}),'parents_with_both_repair_and_regression':len({normalize_parent(s) for s in repairs}&{normalize_parent(s) for s in regressions}),'remaining_errors':sum(q['anomaly']!=q['predicted_anomaly'] for q in rr)})
 return per,summaries
PROTOCOL=read(W/'protocol.json');CATS=PROTOCOL['categories'];SETTINGS=['uniform_mask','quality_select']
def normal():
 assert not (OUT/'normal_lock.json').exists(),'Normal phase frozen; use a new experiment for changes.'
 protected=read(W/'protected_before.json')
 for p,h in protected.items():assert sha(p)==h,p
 mi={r['source_id']:r for r in read(W/'inputs/export_manifest.json')['records']};zi=mask_index('normal');cache={};qualities=[]
 for cat in CATS:
  rr=rows(HY/'variants/evidence_edit/readers'/cat/'normal_facts.jsonl');cache[cat]=[]
  for r in rr:
   ev=candidate_record(cat,r,get_mask(cat,r,zi,mi));cache[cat].append((r,ev))
   if r['split']=='dev':qualities += [e['mask']['quality'] for e in ev if e['readable']]
  print('normal candidates',cat,len(rr),flush=True)
 floor=float(np.quantile(qualities,.10));dump(OUT/'quality_floor.json',{'floor':floor,'quantile':.10,'readable_normal_DEV_target_axes':len(qualities),'DEV_quality_quantiles':np.quantile(qualities,[0,.05,.1,.25,.5,.9,1]).tolist(),'test_or_CAL_used':False,'global_not_per_category':True})
 deployment={s:copy.deepcopy(read(HY/'variants/evidence_edit/deployment.json')) for s in SETTINGS}
 for cat,cc in cache.items():
  rr=[r for r,e in cc];original=read(HY/'variants/evidence_edit/deployment.json')[cat]
  assert calibrate(cat,rr,original['aux'])==original,'Original normal calibration reconstruction failed: '+cat
  save(OUT/'readout_evidence'/cat/'normal.jsonl.gz',[{'source_id':r['source_id'],'sha256':r['sha256'],'split':r['split'],'events':e} for r,e in cc])
  for s in SETTINGS:
   rr=[transform(cat,r,e,s,floor) for r,e in cc];save(OUT/s/(cat+'_normal.jsonl.gz'),rr);deployment[s][cat]=calibrate(cat,rr,original['aux'])
 for s in SETTINGS:dump(OUT/s/'deployment.json',deployment[s])
 inputs=protected.copy()
 for p in [Path(__file__),CODE/'hybrid.py',CODE/'original_executor.py',W/'inputs/export_manifest.json',W/'inputs/frozen_masks.tar.gz',ZL/'normal_masks.npz',ZL/'test_masks.npz',ZL/'normal_axis_evidence.json',ZL/'test_axis_evidence.json']:
  inputs[str(p.resolve())]=sha(p)
 for p in OUT.rglob('*'):
  if p.is_file():inputs[str(p.resolve())]=sha(p)
 dump(OUT/'normal_lock.json',{'state':'frozen_before_new_test_readouts','files':inputs,'zero_new_training':True,'zero_new_forward_calls':True,'original_calibration_reproduced_exactly':True,'quality_floor':floor})
 print('NORMAL LOCKED; global quality floor',floor,flush=True)
def evaluate():
 assert not (OUT/'summary.json').exists(),'Completed results are immutable; do not overwrite.'
 lock=read(OUT/'normal_lock.json')
 for p,h in lock['files'].items():assert sha(p)==h,p
 original=read(HY/'evaluation/evidence_edit/predictions.json');orig={r['source_id']:r for r in original};preds={(s,m):orig.copy() for s in SETTINGS for m in ['own_normal','shared_original']};mi={r['source_id']:r for r in read(W/'inputs/export_manifest.json')['records']};zi=mask_index('test');floor=lock['quality_floor'];configs={s:read(OUT/s/'deployment.json') for s in SETTINGS};origcfg=read(HY/'variants/evidence_edit/deployment.json');caseledger=[];stats={s:collections.Counter() for s in SETTINGS}
 for cat in CATS:
  rr=rows(HY/'variants/evidence_edit/readers'/cat/'test_facts.jsonl');cache=[]
  for r in rr:
   old=orig[r['source_id']];baseline=classify(cat,r,origcfg[cat],old);assert baseline==old,'Original frozen prediction mismatch: '+r['source_id']
   e=candidate_record(cat,r,get_mask(cat,r,zi,mi));cache.append((r,e))
  save(OUT/'readout_evidence'/cat/'test.jsonl.gz',[{'source_id':r['source_id'],'sha256':r['sha256'],'split':r['split'],'events':e} for r,e in cache])
  for s in SETTINGS:
   changed=[]
   for r,e in cache:
    n=transform(cat,r,e,s,floor);changed.append(n);stats[s]['images']+=1;stats[s]['changed_images']+=bool(n['changed_axes']);stats[s]['changed_axes']+=len(n['changed_axes']);stats[s]['target_axes']+=len(e);stats[s]['unreadable_axes']+=sum(not x['readable'] for x in e)
    for mode in ['own_normal','shared_original']:
     cfg=configs[s][cat] if mode=='own_normal' else origcfg[cat];p=classify(cat,n,cfg,orig[r['source_id']]);preds[s,mode][r['source_id']]=p
   save(OUT/s/(cat+'_test.jsonl.gz'),changed)
  print('test replay',cat,len(rr),flush=True)
 summary=[];ledgers=[]
 for s in SETTINGS:
  for mode in ['own_normal','shared_original']:
   pp=[preds[s,mode][r['source_id']] for r in original];save(OUT/s/('predictions_'+mode+'.jsonl.gz'),pp);per,summ=summarize(pp,orig);dump(OUT/s/('per_task_'+mode+'.json'),per)
   for r in summ:summary.append({'setting':s,'reference_mode':mode,**r})
   for p in pp:
    o=orig[p['source_id']]
    if o['predicted_anomaly']!=p['predicted_anomaly']:
     ledgers.append({'setting':s,'reference_mode':mode,'source_id':p['source_id'],'parent_id':normalize_parent(p['source_id']),'dataset':p['dataset'],'category':p['category'],'task':p['task'],'anomaly':p['anomaly'],'transition':('FN_to_TP' if p['anomaly'] else 'FP_to_TN') if p['predicted_anomaly']==p['anomaly'] else ('TP_to_FN' if p['anomaly'] else 'TN_to_FP'),'old_score':o['score'],'old_threshold':o['threshold'],'new_score':p['score'],'new_threshold':p['threshold']})
 baseper,basesumm=summarize(original,orig)
 for r in basesumm:summary.insert(0,{'setting':'original_frozen','reference_mode':'original',**r})
 dump(OUT/'summary.json',summary);dump(OUT/'readout_statistics.json',{s:dict(c) for s,c in stats.items()});dump(OUT/'paired_transitions.json',ledgers)
 for name,rr in [('summary.csv',summary),('paired_transitions.csv',ledgers)]:
  keys=list(dict.fromkeys(k for r in rr for k in r))
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
 for sid in ['Cookies/test/logical_anomalies/Quantity/001.jpg','Balls/test/good/000.jpg','Tapes/test/logical_anomalies/Length/000.jpg','screw_bag/test/good/017.png','screw_bag/test/logical_anomalies/000.png','Stationery_Blurry_CD/test/good/001.jpg']:
  caseledger.append({'source_id':sid,'original':orig[sid],'variants':{s:{mode:preds[s,mode][sid] for mode in ['own_normal','shared_original']} for s in SETTINGS}})
 dump(OUT/'preselected_cases.json',caseledger)
 for p,h in lock['files'].items():assert sha(p)==h,p
 dump(OUT/'completion.json',{'state':'complete','all_original_inputs_unchanged':True,'normal_calibration_reconstructed_exactly':True,'all_affected_original_test_predictions_reconstructed_exactly':True,'test_records_each_setting':len(original),'changed_scope_test_records':sum(len(rows(HY/'variants/evidence_edit/readers'/cat/'test_facts.jsonl')) for cat in CATS),'no_abstention_or_case_dropping':True,'new_training':0,'network_forward_calls':0,'baseline_runs':0,'seconds_since_epoch':time.time()})
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in ['normal','evaluate']
 globals()[sys.argv[1]]()
