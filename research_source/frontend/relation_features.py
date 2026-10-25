"""Measured physical relations with explicit unknowns; normal-only scales/bounds."""
import math
import numpy as np
from common import CATEGORIES,EXPECTED

def geometry(category,facts):
    attrs=facts['attributes'];by={c:attrs.get(c+'_instances',[]) for c in CATEGORIES[category]};out={};unknown=[]
    def margin(point,box):
        x,y=point;x0,y0,x1,y1=box;return min(x-x0,x1-x,y-y0,y1-y)/max(x1-x0,y1-y0,.02)
    if category=='Cookies':
        for cookie,dish,wrong in [('yellow_cookie','square_dish','round_dish'),('black_cookie','round_dish','square_dish')]:
            if by[cookie] and by[dish]:
                correct=[max(margin(o['center'],d['box']) for d in by[dish]) for o in by[cookie]];out[cookie+'_containment_margin']=min(correct)
                if by[wrong]:out[cookie+'_binding_margin']=min(a-max(margin(o['center'],d['box']) for d in by[wrong]) for a,o in zip(correct,by[cookie]))
            else:unknown.append(cookie+'_binding')
    if category=='Dishes':
        if all(by[c] for c in ['fork','plate','spoon']):
            fork,plate,spoon=[by[c][0] for c in ['fork','plate','spoon']];w=max(.02,plate['box'][2]-plate['box'][0]);out['utensil_order_margin']=min(plate['center'][0]-fork['center'][0],spoon['center'][0]-plate['center'][0])/w
        else:unknown.append('utensil_order')
    if category=='Balls':
        if by['orange_ball'] and by['white_ball']:
            out['color_row_order_margin']=min(o['center'][1] for o in by['white_ball'])-max(o['center'][1] for o in by['orange_ball'])
        else:unknown.append('color_row_order')
        if by['transparent_case']:
            case=by['transparent_case'][0];balls=by['orange_ball']+by['white_ball'];box=case['box'];cx,cy=case['center'];slots=[0,0,0,0]
            if balls:out['ball_inside_case_margin']=min(margin(o['center'],box) for o in balls)
            for o in balls:slots[int(o['center'][1]>=cy)*2+int(o['center'][0]>=cx)]+=1
            out['cell_occupancy_error']=float(sum(abs(n-1) for n in slots))
        else:unknown.append('case_coordinate_frame')
    if category=='Ropes':
        text=attrs.get('label_text');color=attrs.get('rope_color');rgb=attrs.get('rope_chromaticity')
        if text in ['RED','GREEN','BLUE'] and color is not None:
            out['label_color_error']=float(text!=color)
            if rgb is not None:
                i=['RED','GREEN','BLUE'].index(text);out['label_color_margin']=float(rgb[i]-max(rgb[j] for j in range(3) if j!=i))
        else:unknown.append('label_color_correspondence')
    if category=='engine_wiring':
        clips=sorted(by['metal_clip'],key=lambda o:o['center'][1]);hoops=by['blue_hoop']
        if len(clips)==2 and len(hoops)==1:
            a=np.array(clips[0]['center']);b=np.array(clips[1]['center']);h=np.array(hoops[0]['center']);v=b-a;t=float(np.dot(h-a,v)/max(1e-8,np.dot(v,v)));out['hoop_between_margin']=min(t,1-t)
        else:unknown.append('hoop_between_clips')
        out.update(facts.get('relation_measurements',{}))
        relation_logits=facts.get('relation_logits',{})
        for key,value in relation_logits.items():out['measured_'+key+'_margin']=float(value)
        if not relation_logits:unknown.extend(facts.get('relation_unknown',[]))
    return out,unknown

def enrich_color(facts,masks,rgb):
    # Measure chromaticity independently of the printed label; low light uses
    # relative chroma rather than a fixed absolute RGB difference of thirty.
    import cv2
    mask=cv2.resize(masks[0].astype('float32'),(rgb.shape[1],rgb.shape[0]))>.5;x=rgb.astype('float32');mx=x.max(2);mn=x.min(2);chosen=mask&(mx>3)&((mx-mn)/np.maximum(mx,1)>.18)
    attrs=facts['attributes']
    if chosen.sum()>=4:
        values=x[chosen];chrom=np.median(values/np.maximum(values.sum(1,keepdims=True),1),axis=0);attrs['rope_chromaticity']=chrom.tolist();attrs['rope_rgb']=np.median(values,axis=0).tolist();attrs['rope_color']=['RED','GREEN','BLUE'][int(np.argmax(chrom))]
    else:attrs['rope_chromaticity']=None;attrs['rope_color']=None

def fit(category,normal):
    train=[r for r in normal if r['split']=='train'];dev=[r for r in normal if r['split']=='dev'];tg=[geometry(category,r['facts'])[0] for r in train];dg=[geometry(category,r['facts'])[0] for r in dev];keys=sorted({k for g in tg for k in g});config={}
    for key in keys:
        tr=[g[key] for g in tg if key in g];dv=[g[key] for g in dg if key in g];scale=max(float(np.median(np.abs(tr))),float(np.quantile(tr,.75)-np.quantile(tr,.25)),.02)
        if key.startswith('contact_') and key.endswith('_u'):
            lo,hi=np.quantile(tr,[.01,.99]);s=max(float(hi-lo),.05);pad=max(.25*s,max([max(lo-v,v-hi,0.) for v in dv],default=0.));config[key]={'kind':'interval','lo':float(lo-pad),'hi':float(hi+pad),'scale':s}
        elif key.endswith('_error'):config[key]={'kind':'error','scale':1.}
        else:config[key]={'kind':'positive_margin','scale':scale}
    required={'Cookies':['yellow_cookie_containment_margin','black_cookie_containment_margin','yellow_cookie_binding_margin','black_cookie_binding_margin'],'Dishes':['utensil_order_margin'],'Balls':['color_row_order_margin','ball_inside_case_margin','cell_occupancy_error'],'Ropes':['label_color_error','label_color_margin'],'engine_wiring':['hoop_between_margin']}
    for key in required.get(category,[]):
        config.setdefault(key,{'kind':'error' if key.endswith('_error') else 'positive_margin','scale':1. if key.endswith('_error') else .1})
    areas={}
    for label in CATEGORIES[category]:
        values=[r.get('mask_areas',{}).get(label) for r in train];values=[v for v in values if v is not None]
        if values:med=float(np.median(values));scale=max(float(np.quantile(values,.75)-np.quantile(values,.25)),med*.1,.001);areas[label]={'median':med,'scale':scale}
    return {'category':category,'geometry':config,'area_reference':areas,'normal_fit_ids':[r['source_id'] for r in train],'development_ids':[r['source_id'] for r in dev],'definition':'Physical margin boundary is zero. Only contact-coordinate intervals and feature units are fitted from normal training/development facts.'}

def verify(category,record,config,with_evidence=False):
    facts=record['facts'];res={label+'_count':float(abs(facts['counts'].get(label,0)-n)) for label,n in zip(CATEGORIES[category],EXPECTED[category])};g,unknown=geometry(category,facts)
    for key,c in config['geometry'].items():
        if key not in g:continue
        v=g[key]
        if c['kind']=='interval':value=max(0.,c['lo']-v,v-c['hi'])/c['scale']
        elif c['kind']=='positive_margin':value=max(0.,-v)/c['scale']
        else:value=max(0.,v)
        res[key]=float(value)
    # Unknown facts are explicitly recorded; they are not fabricated as normal.
    if unknown:res['measurement_unavailable']=1.
    if with_evidence:
        for label,c in config['area_reference'].items():
            value=record.get('mask_areas',{}).get(label)
            if value is not None:res[label+'_area_deficit']=float(max(0.,(c['median']-value)/c['scale']))
    return {'score':float(max(res.values(),default=0.)+.01*sum(res.values())),'residuals':res,'unknown':unknown,'measured_geometry':g}
