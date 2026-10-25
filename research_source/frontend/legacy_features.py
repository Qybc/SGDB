"""Explicit facts and fixed published predicates, plus normal-only geometry."""
from collections import Counter
import math
import numpy as np
from common import CATEGORIES,EXPECTED


def facts_for(category,objects,masks,rgb):
    labels=CATEGORIES[category];counts=Counter(o['class'] for o in objects)
    by={label:[o for o in objects if o['class']==i] for i,label in enumerate(labels)}
    measures={};attributes={}
    for i,label in enumerate(labels):
        oo=by[label]
        # Per-instance axes remain measurements, independent of the normal rule.
        lengths=[float(o.get('measured_axis_length',math.hypot(o['box'][2]-o['box'][0],o['box'][3]-o['box'][1]))) for o in oo]
        if oo:
            measures[label+'_cx']=float(np.mean([o['center'][0] for o in oo]));measures[label+'_cy']=float(np.mean([o['center'][1] for o in oo]))
            measures[label+'_axis_min']=float(min(lengths));measures[label+'_axis_max']=float(max(lengths))
        attributes[label+'_instances']=[{'center':o['center'],'box':o['box'],'axis_length':v,'detector_score':o['score']} for o,v in zip(oo,lengths)]
    if category=='Sticks' and len(by['blue_stick'])==2 and len(by['red_stick'])==1:
        measures['blue_red_length_ratio']=np.mean([x['axis_length'] for x in attributes['blue_stick_instances']])/max(1e-6,attributes['red_stick_instances'][0]['axis_length'])
    if category=='Tapes' and len(by['green_tape'])==len(by['red_tape'])==1:
        measures['green_red_length_ratio']=attributes['green_tape_instances'][0]['axis_length']/max(1e-6,attributes['red_tape_instances'][0]['axis_length'])
    if category=='Stationery':
        for long,short,key in [('black_pencil','red_pencil','pencil_length_ratio'),('blue_eraser','red_eraser','eraser_length_ratio')]:
            if by[long] and by[short]:measures[key]=attributes[long+'_instances'][0]['axis_length']/max(1e-6,attributes[short+'_instances'][0]['axis_length'])
    if category=='Ropes':
        if by['rope'] and by['reference_stick']:
            measures['rope_reference_length_ratio']=attributes['rope_instances'][0]['axis_length']/max(1e-6,attributes['reference_stick_instances'][0]['axis_length'])
        # Color is read from observed rope pixels, not from the label/rule.
        h,w=rgb.shape[:2];sat=rgb.max(2).astype(float)-rgb.min(2).astype(float)
        region=np.zeros((h,w),bool)
        for o in by['rope']:
            x0,y0,x1,y1=[int(v*s) for v,s in zip(o['box'],[w,h,w,h])];region[y0:y1,x0:x1]=True
        # Restrict color to the predicted rope component; a loose box can
        # contain colored background cables that are not the target rope.
        component=masks[0];yi=np.minimum(np.arange(h)*component.shape[0]//h,component.shape[0]-1);xi=np.minimum(np.arange(w)*component.shape[1]//w,component.shape[1]-1)
        region &= component[yi[:,None],xi[None,:]]>.5
        chosen=region&(sat>30)
        if chosen.sum()>3:
            color=rgb[chosen].mean(0);attributes['rope_rgb']=color.tolist();attributes['rope_color']=['RED','GREEN','BLUE'][int(np.argmax(color))]
        else:attributes['rope_color']=None
    if category=='engine_wiring':
        for c,label in enumerate(labels[:3]):
            # Distance from each clip center to its nearby visible wire. The
            # clipped/occluded interior is not filled using an assumed route.
            yy,xx=np.where(masks[c]>.5);points=np.stack([xx/masks.shape[2],yy/masks.shape[1]],axis=1) if len(xx) else np.empty((0,2))
            distances=[]
            for clip in by['metal_clip']:
                center=np.array(clip['center']);distance=float(np.linalg.norm(points-center,axis=1).min()) if len(points) else 1.
                distances.append(distance)
            if distances:measures[label+'_clip_distance_max']=max(distances)
    return {'counts':{label:counts[i] for i,label in enumerate(labels)},'measurements':{k:float(v) for k,v in measures.items()},'attributes':attributes}


def verify(category,facts,continuous=None):
    violations={};labels=CATEGORIES[category]
    for label,number in zip(labels,EXPECTED[category]):violations[label+'_count']=float(abs(facts['counts'][label]-number))
    attrs=facts['attributes'];measures=facts['measurements']
    def centers(label):return [x['center'] for x in attrs[label+'_instances']]
    if category=='Tools':
        for label,lo,hi in [('bolt',0,1/3),('washer',1/3,2/3),('nut',2/3,1)]:
            violations[label+'_bin']=float(sum(not lo<=x<hi for x,y in centers(label)))
    if category=='Blocks':
        for label,lo,hi in [('circle_block',0,1/3),('triangle_block',1/3,2/3),('square_block',2/3,1)]:
            violations[label+'_bin']=float(sum(not lo<=y<hi for x,y in centers(label)))
    if category=='Cookies':
        for cookie,dish in [('yellow_cookie','square_dish'),('black_cookie','round_dish')]:
            dishes=attrs[dish+'_instances']
            violations[cookie+'_dish_pairing']=float(sum(not any(d['box'][0]<=x<=d['box'][2] and d['box'][1]<=y<=d['box'][3] for d in dishes) for x,y in centers(cookie)))
    if category=='Stationery':
        for label,left in [('black_pencil',True),('blue_eraser',True),('red_pencil',False),('red_eraser',False)]:
            violations[label+'_bin']=float(sum((x>=.5) if left else (x<.5) for x,y in centers(label)))
        for eraser,pencil in [('blue_eraser','black_pencil'),('red_eraser','red_pencil')]:
            violations[eraser+'_left_of_pencil']=float(bool(centers(eraser) and centers(pencil) and centers(eraser)[0][0]>=centers(pencil)[0][0]))
    if category=='Dishes':
        order=[centers(x) for x in ['fork','plate','spoon']]
        violations['fork_plate_spoon_order']=float(all(order) and not order[0][0][0]<order[1][0][0]<order[2][0][0])
    if category=='Balls':
        cases=attrs['transparent_case_instances'];middle=(cases[0]['box'][1]+cases[0]['box'][3])/2 if cases else .5
        violations['orange_upper']=float(sum(y>=middle for x,y in centers('orange_ball')));violations['white_lower']=float(sum(y<middle for x,y in centers('white_ball')))
    if category=='engine_wiring':
        clips=centers('metal_clip');hoops=centers('blue_hoop')
        if len(clips)==2 and len(hoops)==1:
            ax,ay=clips[0];bx,by=clips[1];hx,hy=hoops[0];vx,vy=bx-ax,by-ay
            projection=((hx-ax)*vx+(hy-ay)*vy)/max(1e-9,vx*vx+vy*vy)
            violations['hoop_between_clips']=float(not 0<=projection<=1)
    if category=='Ropes':
        text=attrs.get('label_text');color=attrs.get('rope_color')
        if text is not None:violations['label_color_correspondence']=float(text!=color)
        else:violations['label_unreadable']=1.
    for key,config in (continuous or {}).items():
        if key not in measures:continue
        value=measures[key];lo=config['lo'];hi=config['hi'];scale=config['scale']
        violations[key]=float(max(0.,lo-value,value-hi)/max(1e-6,scale))
    score=max(violations.values(),default=0.)+.01*sum(violations.values())
    return {'score':float(score),'residuals':violations,'violations':[k for k,v in violations.items() if v>0]}


def continuous_keys(category):
    return {
      'Sticks':['blue_red_length_ratio'], 'Tapes':['green_red_length_ratio'],
      'Stationery':['pencil_length_ratio','eraser_length_ratio'], 'Ropes':['rope_reference_length_ratio'],
      'pipe_staple':['retaining_staple_cx','retaining_staple_cy'],
      'tank_screw':['target_screw_cx','target_screw_cy'],
      'underbody_screw':['left_screw_cx','left_screw_cy'],
      'engine_wiring':['blue_hoop_cx','blue_hoop_cy','blue_hoop_axis_min','blue_hoop_axis_max','orange_wire_clip_distance_max','green_wire_clip_distance_max','yellow_wire_clip_distance_max'],
    }.get(category,[])
