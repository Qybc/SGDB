"""Local scalar audit of published operands/executor; imports no detector or OGD implementation."""
import copy,json,math,runpy
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[2];OLD=BASE/'outputs/固定方案_FMP-v1_20261019/results'
DEFS=runpy.run_path(str(OLD/'common.py'));LABELS=DEFS['CATEGORIES'];EXPECTED=DEFS['EXPECTED'];VID=DEFS['VID_LABELS']
def q(x,p,higher=False):
 x=sorted(x);a=(len(x)-1)*p;i=math.floor(a)
 return x[math.ceil(a)] if higher else x[i]+(x[min(i+1,len(x)-1)]-x[i])*(a-i)
def close(a,b,tol=1e-9):
 if isinstance(a,dict):assert a.keys()==b.keys(),(a.keys()-b.keys(),b.keys()-a.keys());[close(a[k],b[k],tol) for k in a]
 elif isinstance(a,(list,tuple)):assert len(a)==len(b);[close(x,y,tol) for x,y in zip(a,b)]
 elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isfinite(a) and abs(a-b)<tol,(a,b)
 else:assert a==b,(a,b)
def relation(cat,f):
 a=f['attributes'];by={c:a.get(c+'_instances',[]) for c in LABELS[cat]};r={};unknown=[]
 def margin(pt,box):return min(pt[0]-box[0],box[2]-pt[0],pt[1]-box[1],box[3]-pt[1])/max(box[2]-box[0],box[3]-box[1],.02)
 if cat=='Cookies':
  for name,dish,other in [('yellow_cookie','square_dish','round_dish'),('black_cookie','round_dish','square_dish')]:
   if by[name] and by[dish]:
    correct=[max(margin(o['center'],d['box']) for d in by[dish]) for o in by[name]];r[name+'_containment_margin']=min(correct)
    if by[other]:r[name+'_binding_margin']=min(v-max(margin(o['center'],d['box']) for d in by[other]) for v,o in zip(correct,by[name]))
   else:unknown.append(name+'_binding')
 if cat=='Dishes':
  if all(by[c] for c in ['fork','plate','spoon']):
   fork,plate,spoon=[by[c][0] for c in ['fork','plate','spoon']];r['utensil_order_margin']=min(plate['center'][0]-fork['center'][0],spoon['center'][0]-plate['center'][0])/max(.02,plate['box'][2]-plate['box'][0])
  else:unknown.append('utensil_order')
 if cat=='Balls':
  if by['orange_ball'] and by['white_ball']:r['color_row_order_margin']=min(o['center'][1] for o in by['white_ball'])-max(o['center'][1] for o in by['orange_ball'])
  else:unknown.append('color_row_order')
  if by['transparent_case']:
   case=by['transparent_case'][0];balls=by['orange_ball']+by['white_ball'];slots=[0]*4
   if balls:r['ball_inside_case_margin']=min(margin(o['center'],case['box']) for o in balls)
   for o in balls:slots[2*int(o['center'][1]>=case['center'][1])+int(o['center'][0]>=case['center'][0])]+=1
   r['cell_occupancy_error']=float(sum(abs(n-1) for n in slots))
  else:unknown.append('case_coordinate_frame')
 if cat=='Ropes':
  text,color,rgb=[a.get(k) for k in ['label_text','rope_color','rope_chromaticity']]
  if text in ['RED','GREEN','BLUE'] and color is not None:
   r['label_color_error']=float(text!=color)
   if rgb is not None:
    i=['RED','GREEN','BLUE'].index(text);r['label_color_margin']=rgb[i]-max(rgb[j] for j in range(3) if j!=i)
  else:unknown.append('label_color_correspondence')
 return r,unknown

def raw(cat,f):
 counts=f['counts'];m=dict(f['measurements']);a=f.get('attributes',{});rr={};unknown=list(f.get('unknown',[]))
 if cat in EXPECTED:rr={label+'_count':float(abs(counts.get(label,0)-n)) for label,n in zip(LABELS[cat],EXPECTED[cat])}
 if cat in ['breakfast_box','juice_bottle','pushpins']:
  exp={'breakfast_box':{'tangerine':2,'peach':1},'juice_bottle':{'bottle':1,'juice':1,'image_label':1,'text_label':1},'pushpins':{'pushpin':15,'cell':15}}[cat];rr.update({k+'_count':float(abs(counts[k]-n)) for k,n in exp.items()})
  if cat=='breakfast_box':rr.update({k+'_presence':float(counts[k]==0) for k in ['oatmeal','dried_fruits','nut','banana_chips']})
  if cat=='juice_bottle':
   rr['fruit_icon_count']=float(abs(f['fruit_icon_count']-1));x,y=f.get('liquid_type'),f.get('fruit_icon_type')
   if x is None or y is None:unknown.append('juice_icon_correspondence')
   else:rr['juice_icon_correspondence']=float(x!=y)
  if cat=='pushpins':rr.update(one_pin_per_cell=float(max((abs(n-1) for n in f['pins_per_detected_cell']),default=1)),pins_outside_cells=float(f['pins_outside_cells']))
 if cat=='screw_bag':rr.update({k+'_count':float(abs(counts[k]-2)) for k in LABELS[cat]});m={k:v for k,v in m.items() if k in ['short_screw_length','long_screw_length']}
 if cat=='splicing_connectors':
  rr.update(terminal_count=float(abs(counts['terminal']-2)),cable_count=float(abs(counts['cable']-1)));expected={'yellow':2,'blue':3,'red':5}.get(f['cable_color']);blocks=f['blocks']
  if expected is None:unknown.append('cable_color')
  else:rr['color_port_count']=float(max((abs(len(b['ports'])-expected) for b in blocks),default=expected))
  if len(blocks)==2 and all(b['occupied_row'] is not None for b in blocks):rr['port_correspondence']=float(abs(blocks[0]['occupied_row']-blocks[1]['occupied_row']))
  else:unknown.append('occupied_port')
  facing=[]
  for i,b in enumerate(blocks):
   if b['entry'] is not None and b['ports']:
    delta=b['entry'][0]-sum(p['center'][0] for p in b['ports'])/len(b['ports']);facing.append(max(0.,-delta if i==0 else delta)/.1)
  rr['entry_facing']=max(facing,default=1.);return rr,{'cable_length_'+str(f['cable_color']):f['cable_length']} if f['cable_length'] is not None else {},unknown
 def centers(label):return [o['center'] for o in a.get(label+'_instances',[])]
 if cat in ['Tools','Blocks']:
  axis=0 if cat=='Tools' else 1
  for label,(lo,hi) in zip(LABELS[cat],[(0,1/3),(1/3,2/3),(2/3,1)]):rr[label+'_bin']=float(sum(not lo<=p[axis]<hi for p in centers(label)))
 if cat=='Stationery':
  for label,left in [('black_pencil',True),('blue_eraser',True),('red_pencil',False),('red_eraser',False)]:rr[label+'_bin']=float(sum(x>=.5 if left else x<.5 for x,y in centers(label)))
  for e,p in [('blue_eraser','black_pencil'),('red_eraser','red_pencil')]:
   if centers(e) and centers(p):rr[e+'_left_of_pencil']=float(centers(e)[0][0]>=centers(p)[0][0])
 for k,v in f.get('relation_features',{}).items():rr[k]=float(max(0.,v if k.endswith('_error') else -v))
 if cat in VID:
  keep={'Sticks':['blue_red_length_ratio'],'Tapes':['green_red_length_ratio'],'Stationery':['pencil_length_ratio','eraser_length_ratio'],'Ropes':['rope_reference_length_ratio']}.get(cat,[]);m={k:v for k,v in m.items() if k in keep}
 return rr,{k:float(v) for k,v in m.items() if v is not None},unknown

def apply_aux(f,model):
 if not model:return f
 f=copy.deepcopy(f);x=f.get('liquid_colour_descriptor')
 if x is None:f.update(liquid_type=None,liquid_type_probabilities=None);return f
 z=np.asarray(model['coef'])@((np.array(x)-model['mean'])/model['scale'])+model['intercept'];p=np.exp(z-z.max());p/=p.sum();f.update(liquid_type=model['classes'][int(p.argmax())],liquid_type_probabilities=p.tolist());return f

def residual(cat,f,config):
 rr,m,unknown=raw(cat,apply_aux(f,config.get('aux')))
 for key,i in config['intervals'].items():
  if key not in m:
   if key.startswith('cable_length_') and not key.endswith(str(f.get('cable_color'))):continue
   unknown.append(key)
  else:rr[key]=max(0.,i['lo']-m[key],m[key]-i['hi'])/i['scale']
 rr['measurement_unavailable']=float(bool(unknown));return rr,sorted(set(unknown))
def score(cat,f,c,task):
 rr,u=residual(cat,f,c);nn={}
 for key,x in rr.items():
  ref=c['references'][task].get(key,[0.]);tail=(1+sum(v>=x for v in ref))/(len(ref)+1);scale=max(q(ref,.75)-q(ref,.25),1.);nn[key]=-math.log(tail)+max(0.,x-max(ref))/scale
 return max(nn.values(),default=0.)+.01*sum(nn.values()),rr,u

def fused(e):
 a=e['anchor'];alpha=e['alpha'];out=copy.deepcopy(a)
 for old,new,quality,r in zip(a,e['local'],e['binding_quality'],out):
  if not alpha or quality<.5:continue
  for k in ['center','box','axis']:r[k]=((1-alpha)*np.array(old[k])+alpha*np.array(new[k])).tolist()
  if r['box'][2]<=r['box'][0] or r['box'][3]<=r['box'][1]:r.update(old)
 return out

def expected_facts(cat,old,objects,size,alpha):
 f=copy.deepcopy(old)
 if not alpha:return f
 m=f['measurements'];a=f.get('attributes')
 if a is not None:
  for i,name in enumerate(LABELS[cat]):
   oo=[o for o in objects if o['class']==i];ii=a[name+'_instances'];assert len(ii)==len(oo);lengths=[]
   for dst,src in zip(ii,oo):
    p,b=src['axis'];length=math.hypot((p[0]-b[0])*size[0],(p[1]-b[1])*size[1])/math.hypot(*size);lengths.append(length);dst.update(center=src['center'],box=src['box'],axis_length=length)
   if oo:m.update({name+'_cx':sum(o['center'][0] for o in oo)/len(oo),name+'_cy':sum(o['center'][1] for o in oo)/len(oo),name+'_axis_min':min(lengths),name+'_axis_max':max(lengths)})
  pairs={'Sticks':[('blue_stick','red_stick','blue_red_length_ratio')],'Tapes':[('green_tape','red_tape','green_red_length_ratio')],'Stationery':[('black_pencil','red_pencil','pencil_length_ratio'),('blue_eraser','red_eraser','eraser_length_ratio')],'Ropes':[('rope','reference_stick','rope_reference_length_ratio')]}.get(cat,[])
  for n,d,k in pairs:
   aa,bb=a[n+'_instances'],a[d+'_instances']
   ok=bool(aa and bb)
   if cat=='Sticks':ok=len(aa)==2 and len(bb)==1
   if cat=='Tapes':ok=len(aa)==len(bb)==1
   if ok:m[k]=(sum(x['axis_length'] for x in aa)/len(aa) if cat=='Sticks' else aa[0]['axis_length'])/max(1e-6,bb[0]['axis_length'])
  f['relation_features'],f['unknown']=relation(cat,f)
 if cat=='screw_bag':
  lengths=sorted(math.hypot((o['axis'][1][0]-o['axis'][0][0])*size[0],(o['axis'][1][1]-o['axis'][0][1])*size[1]) for o in objects if o['class']==2);f['screw_lengths']=lengths
  if len(lengths)==2:m.update(short_screw_length=lengths[0],long_screw_length=lengths[1])
 if cat=='breakfast_box':m['fruit_rightmost_center']=max((o['center'][0] for o in objects if o['class'] in [2,3]),default=None)
 if cat=='pushpins':
  cells=[o for o in objects if o['class']==1];pins=[o for o in objects if o['class']==0];counts=[0]*len(cells);outside=0
  for pin in pins:
   possible=[]
   for i,cell in enumerate(cells):
    x,y=pin['center'];x0,y0,x1,y1=cell['box'];w=x1-x0;h=y1-y0
    if x0-.03*w<=x<=x1+.03*w and y0-.03*h<=y<=y1+.03*h:possible.append((((x-cell['center'][0])/max(.001,w))**2+((y-cell['center'][1])/max(.001,h))**2,i))
   if possible:counts[min(possible)[1]]+=1
   else:outside+=1
  f.update(pins_per_detected_cell=counts,pins_outside_cells=outside)
 if cat=='splicing_connectors':
  terminals=sorted([o for o in objects if o['class']==0],key=lambda o:o['center'][0]);ports=[o for o in objects if o['class']==2];entries=[b['entry'] for b in old['blocks']];blocks=[]
  for i,t in enumerate(terminals):
   pp=[]
   for p in ports:
    distances=[math.hypot(p['center'][0]-tt['center'][0],p['center'][1]-tt['center'][1]) for tt in terminals]
    if min(range(len(terminals)),key=lambda j:distances[j])==i:pp.append(p)
   pp.sort(key=lambda p:p['center'][1]);entry=entries[min(i,len(entries)-1)] if entries else None
   occupied=min(range(len(pp)),key=lambda j:abs(pp[j]['center'][1]-entry[1])) if pp and entry else None;blocks.append({'ports':pp,'occupied_row':occupied,'entry':entry,'terminal':t})
  f['blocks']=blocks
 return f
