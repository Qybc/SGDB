"""Frozen neural identities, one-to-one evidence association, frozen geometric witnesses.

No network forward, optimizer, expected count, or anomaly label is consumed here.
Unbound identities count as recognized, but never acquire invented geometric values.
"""
import copy, math
import numpy as np
from scipy.optimize import linear_sum_assignment

CANDIDATES = [{'name':f'{kind}_r{radius}', 'class_aware':kind=='same_class', 'radius':radius}
              for kind in ['same_class','any_class'] for radius in [.04,.08]]

def bind(decoded, witnesses, scheme):
    n,m=len(decoded),len(witnesses)
    if not n:return [],[]
    # Per-query dummy permits genuine non-matches without forcing a distant pair.
    cost=np.full((n,m+n),scheme['radius']+1e-7,dtype=float)
    distance=np.zeros((n,m))
    for i,o in enumerate(decoded):
        for j,a in enumerate(witnesses):
            d=math.dist(o['center'],a['center']);distance[i,j]=d
            permitted=(not scheme['class_aware'] or o['class']==a['class'])
            cost[i,j]=d if permitted and d<=scheme['radius'] else 1e6
    ii,jj=linear_sum_assignment(cost);pairs={int(i):int(j) for i,j in zip(ii,jj) if j<m and cost[i,j]<1e5}
    objects=[];trace=[]
    for i,o in enumerate(decoded):
        j=pairs.get(i)
        if j is None:
            objects.append({'class':o['class'],'score':o['score'],'center':None,'box':None,'axis':None})
            trace.append({'decoded_index':i,'witness_index':None,'distance':None})
        else:
            a=witnesses[j]
            objects.append({'class':o['class'],'score':o['score'],**{k:copy.deepcopy(a[k]) for k in ['center','box','axis']}})
            trace.append({'decoded_index':i,'witness_index':j,'distance':float(distance[i,j])})
    assert len({t['witness_index'] for t in trace if t['witness_index'] is not None})==len(pairs)
    return objects,trace

def facts(cat, source, objects, labels, size, relation):
    """Retain cached pixel/OCR witnesses; rebuild all object-derived operands."""
    f=copy.deepcopy(source);counts={c:sum(o['class']==i for o in objects) for i,c in enumerate(labels)}
    bound=[o for o in objects if o['center'] is not None];f['counts']=counts
    missing=[c for i,c in enumerate(labels) if any(o['class']==i and o['center'] is None for o in objects)]
    unknown=['unbound_geometry_'+c for c in missing]
    m=f['measurements'];attrs=f.get('attributes')
    if attrs is not None:
        # Appearance, color and OCR are frozen cached pixel observations.
        for i,c in enumerate(labels):
            for suffix in ['_cx','_cy','_axis_min','_axis_max']:m.pop(c+suffix,None)
            oo=[o for o in bound if o['class']==i]
            aa=[{'center':o['center'],'box':o['box'],
                 'axis_length':float(np.linalg.norm((np.array(o['axis'][1])-o['axis'][0])*size)/np.linalg.norm(size)),
                 'detector_score':o['score']} for o in oo]
            attrs[c+'_instances']=aa
            if aa and c not in missing:
                ll=[a['axis_length'] for a in aa]
                m.update({c+'_cx':float(np.mean([a['center'][0] for a in aa])),c+'_cy':float(np.mean([a['center'][1] for a in aa])),c+'_axis_min':min(ll),c+'_axis_max':max(ll)})
        pairs={'Sticks':[('blue_stick','red_stick','blue_red_length_ratio')],
               'Tapes':[('green_tape','red_tape','green_red_length_ratio')],
               'Stationery':[('black_pencil','red_pencil','pencil_length_ratio'),('blue_eraser','red_eraser','eraser_length_ratio')],
               'Ropes':[('rope','reference_stick','rope_reference_length_ratio')]}.get(cat,[])
        for a,b,key in pairs:
            m.pop(key,None);aa=attrs[a+'_instances'];bb=attrs[b+'_instances'];ok=bool(aa and bb) and not ({a,b}&set(missing))
            if cat=='Sticks':ok=ok and len(aa)==2 and len(bb)==1
            if cat=='Tapes':ok=ok and len(aa)==len(bb)==1
            if ok:m[key]=(float(np.mean([x['axis_length'] for x in aa])) if cat=='Sticks' else aa[0]['axis_length'])/max(1e-6,bb[0]['axis_length'])
        rr,u=relation(cat,f);f['relation_features']=rr;unknown+=u
    if cat=='screw_bag':
        m.pop('short_screw_length',None);m.pop('long_screw_length',None)
        lengths=sorted(float(np.linalg.norm((np.array(o['axis'][1])-o['axis'][0])*size)) for o in bound if o['class']==2)
        f['screw_lengths']=lengths
        if len(lengths)==2 and labels[2] not in missing:m.update(short_screw_length=lengths[0],long_screw_length=lengths[1])
    if cat=='breakfast_box':
        m['nut_visible_count']=counts['nut'];m['banana_visible_count']=counts['banana_chips']
        m['fruit_rightmost_center']=None if {'tangerine','peach'}&set(missing) else max((o['center'][0] for o in bound if o['class'] in [2,3]),default=None)
    if cat=='juice_bottle':
        icons=sorted([o for o in objects if o['class']>=4],key=lambda o:-o['score']);chosen=[]
        def iou(a,b):
            ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]));z=ix*iy
            return z/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z)
        for o in icons:
            if o['box'] is None or not any(p['box'] is not None and iou(o['box'],p['box'])>.5 for p in chosen):chosen.append(o)
        f['fruit_icon_count']=len(chosen);f['fruit_icon_type']=labels[chosen[0]['class']] if chosen else None
    if cat=='pushpins':
        cells=[o for o in bound if o['class']==1];pins=[o for o in bound if o['class']==0];assigned=[0]*len(cells);outside=0
        for p in pins:
            possible=[]
            for i,c in enumerate(cells):
                a,b,d,e=c['box'];x,y=p['center'];w,h=d-a,e-b
                if a-.03*w<=x<=d+.03*w and b-.03*h<=y<=e+.03*h:possible.append((((x-c['center'][0])/max(.001,w))**2+((y-c['center'][1])/max(.001,h))**2,i))
            if possible:assigned[min(possible)[1]]+=1
            else:outside+=1
        f.update(pins_per_detected_cell=assigned,pins_outside_cells=outside)
        # Separator contrast remains the frozen FMP image witness; no new image read.
    if cat=='splicing_connectors':
        terminals=sorted([o for o in bound if o['class']==0],key=lambda o:o['center'][0]);ports=[o for o in bound if o['class']==2]
        entries=[b['entry'] for b in source['blocks']];blocks=[]
        for i,t in enumerate(terminals):
            pp=[p for p in ports if min(range(len(terminals)),key=lambda j:math.dist(p['center'],terminals[j]['center']))==i]
            pp.sort(key=lambda o:o['center'][1]);entry=entries[min(i,len(entries)-1)] if entries else None
            occupied=min(range(len(pp)),key=lambda j:abs(pp[j]['center'][1]-entry[1])) if pp and entry is not None else None
            blocks.append({'ports':pp,'occupied_row':occupied,'entry':entry,'terminal':t})
        f['blocks']=blocks
    f['unknown']=sorted(set(unknown))
    return f
