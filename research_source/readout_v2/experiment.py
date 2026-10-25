"""Frozen follow-up: require complete, exclusive support before replacing an axis."""
from pathlib import Path
import collections,copy,csv,gzip,hashlib,importlib.util,json,math,sys,time
import numpy as np
from scipy import ndimage
W=Path(__file__).resolve().parent
V1=W.parent/'readout_v1'
sp=importlib.util.spec_from_file_location('frozen_measurement_v1',V1/'experiment.py');v1=importlib.util.module_from_spec(sp);sp.loader.exec_module(v1)
HY=v1.HY;OUT=W/'results';read=v1.read;sha=v1.sha;rows=v1.rows;dump=v1.dump;save=v1.save
P=read(W/'protocol.json');CATS=P['categories'];SETTINGS=P['new_settings'];FLOOR=read(W.parents[1]/'configs/quality_floor.json')['floor']

def extra_checks(cat,r,masks,prior_events):
    size=np.array(r['native_size'],float)
    events=copy.deepcopy(prior_events);by_index={e['object_index']:e for e in events}
    for cls,mask in masks.items():
        ids=[i for i,o in enumerate(r['objects']) if o['class']==cls and o['center'] is not None and o['axis'] is not None]
        if not ids:continue
        h,w=mask.shape;yy,xx=np.where(mask);points=np.column_stack([(xx+.5)/w,(yy+.5)/h]);native=points*size
        centers=np.array([r['objects'][i]['center'] for i in ids])*size
        owner=np.argmin(((native[:,None,:]-centers[None,:,:])**2).sum(2),axis=1) if len(points) else np.empty(0,int)
        lab,n=ndimage.label(mask,structure=np.ones((3,3)));areas=np.bincount(lab.ravel());minimum=max(16,int(math.ceil(.02*len(points))))
        substantial={i for i in range(1,n+1) if areas[i]>=minimum}
        group=[]
        for j,i in enumerate(ids):
            e=by_index[i];assert len(native[owner==j])==e['assigned_pixels']
            eligible=e['readable'] and e['mask']['quality']>=FLOOR and e['mask']['quality']>=e['old']['quality']+.05
            labels=lab[yy[owner==j],xx[owner==j]];count=np.bincount(labels,minlength=n+1);dom=int(np.argmax(count[1:])+1) if n and len(labels) else None
            recovered=float(count[dom]/areas[dom]) if dom is not None else 0.
            comp=lab==dom if dom is not None else np.zeros(mask.shape,bool)
            cy,cx=np.where(comp);anchor=np.floor(np.array(r['objects'][i]['center'])*np.array([w,h])).astype(int)
            near=ndimage.binary_dilation(comp,structure=np.ones((3,3)),iterations=2)
            anchor_ok=bool(0<=anchor[0]<w and 0<=anchor[1]<h and near[anchor[1],anchor[0]])
            border=bool(len(cx) and (cx.min()<=1 or cy.min()<=1 or cx.max()>=w-2 or cy.max()>=h-2))
            other_boxes=[]
            for k,o in enumerate(r['objects']):
                if k!=i and o['box'] is not None:
                    b=np.array(o['box'],float);b[:2]-=np.array([1/w,1/h]);b[2:]+=np.array([1/w,1/h]);other_boxes.append(b)
            occluded=False;occluders=[];removed_samples=0;shortened=False
            if e['readable']:
                a=np.array(e['old_axis'])*size;b=np.array(e['mask_axis'])*size
                old_length=float(np.linalg.norm(a[1]-a[0]));new_length=float(np.linalg.norm(b[1]-b[0]));pixel=float(np.linalg.norm(size/np.array([w,h])))
                shortened=new_length+pixel<old_length
                if shortened:
                    direction=(b[1]-b[0])/new_length;samples=a[0]+np.linspace(0,1,64)[:,None]*(a[1]-a[0]);proj=(samples-b[0])@direction
                    removed=samples[(proj < -pixel)|(proj > new_length+pixel)];removed_samples=len(removed)
                    probes=np.vstack([b,removed])/size
                    for k,o in enumerate(r['objects']):
                        if k==i or o['box'] is None:continue
                        box=np.array(o['box'],float);box[:2]-=np.array([1/w,1/h]);box[2:]+=np.array([1/w,1/h])
                        if np.any(np.all((probes>=box[:2])&(probes<=box[2:]),axis=1)):occluders.append(k)
                    occluded=bool(occluders)
            e['v1_eligible']=bool(eligible)
            e['completeness']={'bound_instances_in_class':len(ids),'substantial_components':len(substantial),'component_minimum_pixels':minimum,'dominant_component_id':dom,'dominant_component_is_substantial':dom in substantial,'dominant_component_recovered_fraction':recovered,'anchor_supported':anchor_ok,'component_touches_image_border_band':border,'shortened_more_than_one_pixel_diagonal':shortened,'removed_old_axis_samples':removed_samples,'potential_occluder_indices':occluders,'occlusion_tail_risk':occluded}
            group.append(e)
        doms=[e['completeness']['dominant_component_id'] for e in group]
        shared=len(set(doms))!=len(doms)
        structural=[]
        if len(substantial)!=len(ids):structural.append('component_count_mismatch')
        if shared:structural.append('shared_dominant_component')
        for e in group:
            f=e['completeness']
            if not f['dominant_component_is_substantial']:structural.append('no_substantial_dominant_component')
            if f['dominant_component_recovered_fraction']<.98:structural.append('partial_component_ownership')
            if not f['anchor_supported']:structural.append('anchor_support_missing')
        structural=sorted(set(structural))
        for e in group:
            e['whole_component_rejection_reasons']=structural
            e['whole_component_pass']=not structural
            extra=[]
            if e['completeness']['component_touches_image_border_band']:extra.append('image_boundary_truncation')
            if e['completeness']['occlusion_tail_risk']:extra.append('occlusion_tail_risk')
            e['complete_support_rejection_reasons']=sorted(set(structural+extra))
            e['complete_support_pass']=not (structural or extra)
    return events

def transform(cat,r,events,setting):
    objects=copy.deepcopy(r['objects']);changed=[]
    for e in events:
        use=e['v1_eligible'] and e[setting+'_pass']
        if use:objects[e['object_index']]['axis']=e['mask_axis'];changed.append(e['object_index'])
    f=v1.facts(cat,r['facts'],objects,v1.LABELS[cat],np.array(r['native_size']),v1.relation)
    assert f['counts']==r['facts']['counts']
    return {**{k:r[k] for k in ['source_id','sha256','split','task','category','dataset','native_size','association']},'objects':objects,'facts':f,'changed_axes':changed}

def diagnostics(cat,rr):
    result=[]
    for role in sorted({r['split'] for r,e in rr}):
        es=[e for r,ev in rr if r['split']==role for e in ev];cnt=collections.Counter()
        for e in es:
            cnt['target_axes']+=1;cnt['v1_eligible_axes']+=e['v1_eligible'];cnt['whole_component_accepted_axes']+=e['v1_eligible'] and e['whole_component_pass'];cnt['complete_support_accepted_axes']+=e['v1_eligible'] and e['complete_support_pass']
            if e['v1_eligible']:
                for reason in e['complete_support_rejection_reasons']:cnt['reject_'+reason]+=1
        result.append({'category':cat,'split':role,**dict(cnt)})
    return result

def normal():
    assert not (OUT/'normal_lock.json').exists(),'Normal phase already frozen.'
    protected=read(W/'protected_before.json')['files']
    for p,h in protected.items():assert sha(p)==h,p
    cfg0=read(HY/'variants/evidence_edit/deployment.json');cfgs={s:copy.deepcopy(cfg0) for s in SETTINGS}
    mi={r['source_id']:r for r in read(V1/'inputs/export_manifest.json')['records']};zi=v1.mask_index('normal');diags=[]
    for cat in CATS:
        rr=rows(HY/'variants/evidence_edit/readers'/cat/'normal_facts.jsonl');old_events={r['source_id']:r for r in rows(V1/'results/readout_evidence'/cat/'normal.jsonl.gz')};cache=[]
        assert v1.calibrate(cat,rr,cfg0[cat]['aux'])==cfg0[cat]
        for r in rr:
            ev=old_events[r['source_id']];assert ev['sha256']==r['sha256']
            cache.append((r,extra_checks(cat,r,v1.get_mask(cat,r,zi,mi),ev['events'])))
        save(OUT/'readout_evidence'/cat/'normal.jsonl.gz',[{'source_id':r['source_id'],'sha256':r['sha256'],'split':r['split'],'events':e} for r,e in cache])
        diags+=diagnostics(cat,cache)
        for s in SETTINGS:
            transformed=[transform(cat,r,e,s) for r,e in cache];save(OUT/s/(cat+'_normal.jsonl.gz'),transformed);cfgs[s][cat]=v1.calibrate(cat,transformed,cfg0[cat]['aux'])
        print('normal completeness',cat,len(rr),flush=True)
    for s in SETTINGS:dump(OUT/s/'deployment.json',cfgs[s])
    dump(OUT/'normal_diagnostics.json',diags)
    inputs=protected.copy();inputs[str(Path(__file__).resolve())]=sha(__file__)
    for p in OUT.rglob('*'):
        if p.is_file():inputs[str(p.resolve())]=sha(p)
    dump(OUT/'normal_lock.json',{'state':'frozen_before_v2_test_readout_decisions','files':inputs,'inherited_quality_floor':FLOOR,'normal_calibration_reproduced_exactly':True,'new_neural_training':0,'new_model_forward_calls':0,'new_hyperparameter_search':False})
    print('NORMAL LOCKED',json.dumps([r for r in diags if r['split']=='dev']),flush=True)

def evaluate():
    assert not (OUT/'summary.json').exists(),'Completed experiment cannot be overwritten.'
    lock=read(OUT/'normal_lock.json')
    for p,h in lock['files'].items():assert sha(p)==h,p
    origrows=read(HY/'evaluation/evidence_edit/predictions.json');orig={r['source_id']:r for r in origrows};original_cfg=read(HY/'variants/evidence_edit/deployment.json');cfgs={s:read(OUT/s/'deployment.json') for s in SETTINGS};preds={(s,m):orig.copy() for s in SETTINGS for m in ['shared_original','own_normal']}
    mi={r['source_id']:r for r in read(V1/'inputs/export_manifest.json')['records']};zi=v1.mask_index('test');diags=[]
    for cat in CATS:
        rr=rows(HY/'variants/evidence_edit/readers'/cat/'test_facts.jsonl');olde={r['source_id']:r for r in rows(V1/'results/readout_evidence'/cat/'test.jsonl.gz')};cache=[]
        for r in rr:
            assert v1.classify(cat,r,original_cfg[cat],orig[r['source_id']])==orig[r['source_id']]
            assert r['sha256']==olde[r['source_id']]['sha256']
            cache.append((r,extra_checks(cat,r,v1.get_mask(cat,r,zi,mi),olde[r['source_id']]['events'])))
        save(OUT/'readout_evidence'/cat/'test.jsonl.gz',[{'source_id':r['source_id'],'sha256':r['sha256'],'split':r['split'],'events':e} for r,e in cache]);diags+=diagnostics(cat,cache)
        for s in SETTINGS:
            nn=[]
            for r,e in cache:
                n=transform(cat,r,e,s);nn.append(n)
                for mode in ['shared_original','own_normal']:
                    cfg=original_cfg[cat] if mode=='shared_original' else cfgs[s][cat];preds[s,mode][r['source_id']]=v1.classify(cat,n,cfg,orig[r['source_id']])
            save(OUT/s/(cat+'_test.jsonl.gz'),nn)
        print('test completeness',cat,len(rr),flush=True)
    summary=[];ledgers=[];category=[]
    for s in SETTINGS:
        for mode in ['shared_original','own_normal']:
            pp=[preds[s,mode][r['source_id']] for r in origrows];save(OUT/s/('predictions_'+mode+'.jsonl.gz'),pp);per,summ=v1.summarize(pp,orig);dump(OUT/s/('per_task_'+mode+'.json'),per)
            summary += [{'setting':s,'reference_mode':mode,**r} for r in summ]
            for cat in CATS:
                subset=[r for r in pp if r['category']==cat];cc=collections.Counter()
                for q in subset:
                    o=orig[q['source_id']];oc=o['anomaly']==o['predicted_anomaly'];nc=q['anomaly']==q['predicted_anomaly'];cc['original_errors']+=not oc;cc['remaining_errors']+=not nc
                    if not oc and nc:cc['repaired']+=1;cc['FN_to_TP' if q['anomaly'] else 'FP_to_TN']+=1
                    if oc and not nc:cc['regressed']+=1;cc['TP_to_FN' if q['anomaly'] else 'TN_to_FP']+=1
                category.append({'setting':s,'reference_mode':mode,'category':cat,'n':len(subset),**{k:cc[k] for k in ['original_errors','repaired','regressed','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP','remaining_errors']},'net_repaired':cc['repaired']-cc['regressed']})
            for q in pp:
                o=orig[q['source_id']]
                if o['predicted_anomaly']!=q['predicted_anomaly']:
                    ledgers.append({'setting':s,'reference_mode':mode,'source_id':q['source_id'],'parent_id':v1.normalize_parent(q['source_id']),'dataset':q['dataset'],'category':q['category'],'task':q['task'],'anomaly':q['anomaly'],'transition':('FN_to_TP' if q['anomaly'] else 'FP_to_TN') if q['anomaly']==q['predicted_anomaly'] else ('TP_to_FN' if q['anomaly'] else 'TN_to_FP'),'old_score':o['score'],'old_threshold':o['threshold'],'new_score':q['score'],'new_threshold':q['threshold']})
    retain=[]
    for mode in ['shared_original','own_normal']:
        prior={r['source_id']:r for r in rows(V1/'results/quality_select'/('predictions_'+mode+'.jsonl.gz'))}
        for s in SETTINGS:
            for cat in CATS:
                ids=[sid for sid,o in orig.items() if o['category']==cat];repairs={sid for sid in ids if orig[sid]['anomaly']!=orig[sid]['predicted_anomaly'] and prior[sid]['anomaly']==prior[sid]['predicted_anomaly']};harm={sid for sid in ids if orig[sid]['anomaly']==orig[sid]['predicted_anomaly'] and prior[sid]['anomaly']!=prior[sid]['predicted_anomaly']};new=preds[s,mode]
                correct=lambda sid:new[sid]['anomaly']==new[sid]['predicted_anomaly']
                retain.append({'setting':s,'reference_mode':mode,'category':cat,'v1_repairs':len(repairs),'v1_repairs_retained':sum(correct(sid) for sid in repairs),'v1_regressions':len(harm),'v1_regressions_prevented':sum(correct(sid) for sid in harm),'new_regressions_not_in_v1':sum(orig[sid]['anomaly']==orig[sid]['predicted_anomaly'] and not correct(sid) and sid not in harm for sid in ids)})
    dump(OUT/'summary.json',summary);dump(OUT/'per_category.json',category);dump(OUT/'paired_transitions.json',ledgers);dump(OUT/'retention_vs_v1.json',retain);dump(OUT/'test_diagnostics.json',diags)
    for name,rr in [('summary',summary),('per_category',category),('paired_transitions',ledgers),('retention_vs_v1',retain)]:
        keys=list(dict.fromkeys(k for r in rr for k in r))
        with (OUT/(name+'.csv')).open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rr)
    caseids=['Tapes/test/logical_anomalies/Length/000.jpg','Tapes_Cable_BG/test/logical_anomalies/Type/045.jpg','screw_bag/test/good/017.png','screw_bag/test/logical_anomalies/019.png','screw_bag/test/logical_anomalies/000.png','screw_bag/test/good/025.png','Stationery_Blurry_CD/test/good/001.jpg']
    dump(OUT/'preselected_cases.json',[{'source_id':sid,'original':orig[sid],'variants':{s:{m:preds[s,m][sid] for m in ['shared_original','own_normal']} for s in SETTINGS}} for sid in caseids])
    for p,h in lock['files'].items():assert sha(p)==h,p
    assert not any(n in sys.modules for n in ['torch','tensorflow','transformers'])
    dump(OUT/'completion.json',{'state':'complete','all_original_and_v1_inputs_unchanged':True,'all_original_affected_predictions_reproduced_exactly':True,'test_records_each_setting_mode':9031,'affected_test_records':3429,'new_training':0,'new_network_forward_calls':0,'new_baseline_runs':0,'new_mask_measurement_candidates':0,'new_hyperparameter_search':False,'finished_unix':time.time()})
    for r in summary:
        if r['scope']=='full':print(r['setting'],r['reference_mode'],r['dataset'],'repair/regress',r.get('repaired',0),r.get('regressed',0),'AUROC/F1',r['AUROC'],r['F1'],flush=True)
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['normal','evaluate'];globals()[sys.argv[1]]()
