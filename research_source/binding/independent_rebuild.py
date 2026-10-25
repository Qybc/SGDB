"""Second implementation for audit; no hybrid/detector imports."""
import copy,math,statistics
from collections import Counter

def rebuild(category, original, selected, names, wh, relation):
    output=copy.deepcopy(original);tally=Counter(item['class'] for item in selected)
    output['counts']={name:tally[c] for c,name in enumerate(names)}
    located=[a for a in selected if a['box'] is not None]
    missing={names[a['class']] for a in selected if a['box'] is None}
    unknown=['unbound_geometry_'+name for name in missing]
    measurements=output['measurements'];attrs=output.get('attributes')
    if attrs is not None:
        for c,name in enumerate(names):
            for s in ('cx','cy','axis_min','axis_max'):measurements.pop(name+'_'+s,None)
            instances=[]
            for a in located:
                if a['class']!=c:continue
                u,v=a['axis'];length=math.hypot((u[0]-v[0])*wh[0],(u[1]-v[1])*wh[1])/math.hypot(*wh)
                instances.append({'center':a['center'],'box':a['box'],'axis_length':length,'detector_score':a['score']})
            attrs[name+'_instances']=instances
            if instances and name not in missing:
                measurements[name+'_cx']=statistics.mean(a['center'][0] for a in instances)
                measurements[name+'_cy']=statistics.mean(a['center'][1] for a in instances)
                measurements[name+'_axis_min']=min(a['axis_length'] for a in instances)
                measurements[name+'_axis_max']=max(a['axis_length'] for a in instances)
        ratios={'Sticks':[('blue_stick','red_stick','blue_red_length_ratio')],'Tapes':[('green_tape','red_tape','green_red_length_ratio')],
                'Ropes':[('rope','reference_stick','rope_reference_length_ratio')],
                'Stationery':[('black_pencil','red_pencil','pencil_length_ratio'),('blue_eraser','red_eraser','eraser_length_ratio')]}
        for a,b,key in ratios.get(category,[]):
            measurements.pop(key,None);x=attrs[a+'_instances'];y=attrs[b+'_instances']
            legal=bool(x and y) and not ({a,b}&missing)
            if category=='Sticks':legal=legal and (len(x),len(y))==(2,1)
            if category=='Tapes':legal=legal and (len(x),len(y))==(1,1)
            if legal:
                numerator=statistics.mean(z['axis_length'] for z in x) if category=='Sticks' else x[0]['axis_length']
                measurements[key]=numerator/max(1e-6,y[0]['axis_length'])
        output['relation_features'],flags=relation(category,output);unknown+=flags
    if category=='screw_bag':
        for name in ['short_screw_length','long_screw_length']:measurements.pop(name,None)
        ll=[]
        for o in located:
            if o['class']==2:
                a,b=o['axis'];ll.append(math.hypot((a[0]-b[0])*wh[0],(a[1]-b[1])*wh[1]))
        ll.sort();output['screw_lengths']=ll
        if len(ll)==2 and 'screw' not in missing:measurements.update(short_screw_length=ll[0],long_screw_length=ll[1])
    elif category=='breakfast_box':
        measurements['nut_visible_count']=output['counts']['nut'];measurements['banana_visible_count']=output['counts']['banana_chips']
        fruits=[a['center'][0] for a in located if a['class'] in (2,3)]
        measurements['fruit_rightmost_center']=None if {'tangerine','peach'}&missing else max(fruits,default=None)
    elif category=='pushpins':
        cells=[o for o in located if o['class']==1];occupancy=[0 for _ in cells];outside=0
        for p in (o for o in located if o['class']==0):
            choices=[]
            for j,c in enumerate(cells):
                x,y=p['center'];left,top,right,bottom=c['box'];w=right-left;h=bottom-top
                if left-.03*w<=x<=right+.03*w and top-.03*h<=y<=bottom+.03*h:
                    choices.append((((x-c['center'][0])/max(.001,w))**2+((y-c['center'][1])/max(.001,h))**2,j))
            if choices:occupancy[min(choices)[1]]+=1
            else:outside+=1
        output.update(pins_per_detected_cell=occupancy,pins_outside_cells=outside)
    elif category=='juice_bottle':
        chosen=[]
        for a in sorted((o for o in selected if o['class']>=4),key=lambda o:-o['score']):
            overlap=False
            for b in chosen:
                if a['box'] is None or b['box'] is None:continue
                x,y,X,Y=a['box'];u,v,U,V=b['box'];intersection=max(0,min(X,U)-max(x,u))*max(0,min(Y,V)-max(y,v))
                overlap |= intersection/max(1e-12,(X-x)*(Y-y)+(U-u)*(V-v)-intersection)>.5
            if not overlap:chosen.append(a)
        output['fruit_icon_count']=len(chosen);output['fruit_icon_type']=names[chosen[0]['class']] if chosen else None
    elif category=='splicing_connectors':
        terminals=sorted((o for o in located if o['class']==0),key=lambda o:o['center'][0]);ports=[o for o in located if o['class']==2];groups=[[] for _ in terminals]
        for p in ports:
            if terminals:
                j=min(range(len(terminals)),key=lambda j:math.hypot(p['center'][0]-terminals[j]['center'][0],p['center'][1]-terminals[j]['center'][1]));groups[j].append(p)
        entries=[b['entry'] for b in original['blocks']];blocks=[]
        for i,(t,pp) in enumerate(zip(terminals,groups)):
            pp.sort(key=lambda p:p['center'][1]);entry=entries[min(i,len(entries)-1)] if entries else None
            occupied=min(range(len(pp)),key=lambda j:abs(pp[j]['center'][1]-entry[1])) if pp and entry is not None else None
            blocks.append({'ports':pp,'occupied_row':occupied,'entry':entry,'terminal':t})
        output['blocks']=blocks
    output['unknown']=sorted(set(unknown));return output
