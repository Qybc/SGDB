"""Pixel facts independent of the rule values. All branch differences describe objects."""
import math
import cv2,numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from common import *
from legacy_features import facts_for,continuous_keys
from relation_features import geometry,enrich_color
from loco_features import extract as loco_extract,classifier_predict

def contact(f,masks):
 clips=sorted(f['attributes']['metal_clip_instances'],key=lambda o:o['center'][1]);out={};unknown=[]
 if len(clips)!=2:return {},['clip_coordinate_frame']
 for c,label in enumerate(CATEGORIES['engine_wiring'][:3]):
  yy,xx=np.where(masks[c]>.5);yy=yy/masks.shape[1];xx=xx/masks.shape[2]
  for slot,clip in enumerate(clips):
   x0,y0,x1,y1=clip['box'];w=max(x1-x0,.02)
   for name,ya,yb in [('entry',y0-.07,y0+.015),('exit',y1-.015,y1+.07)]:
    chosen=(yy>=ya)&(yy<=yb);key=f'contact_{label}_{slot}_{name}_u'
    if chosen.sum()>=2:out[key]=float((np.median(xx[chosen])-x0)/w)
    else:unknown.append(key)
 return out,unknown

def connector_facts(image,objects,masks):
 terminals=sorted([o for o in objects if o['class']==0],key=lambda o:o['center'][0]);ports=[o for o in objects if o['class']==2];binary=masks[1]>.5;yy,xx=np.where(binary);H,W=binary.shape;rgb=np.array(image.resize((W,H),Image.Resampling.LANCZOS));v=rgb[binary].astype(float);color=None;length=None;entries=[]
 if len(xx)>=4:
  r,g,b=v.T;num={'red':int(((r>100)&(r>2*g)&(r>2*b)).sum()),'blue':int(((b>45)&(b>1.4*r)&(b>1.15*g)).sum()),'yellow':int(((r>100)&(g>85)&(b<.65*np.minimum(r,g))).sum())};color=max(num,key=num.get)
  if num[color]/len(v)<.05:color=None
  sk=skeletonize(binary);length=0.;sx=image.width/W;sy=image.height/H
  for dx,dy in [(1,0),(0,1),(1,1),(-1,1)]:
   aa=sk[max(0,dy):H+min(0,dy),max(0,dx):W+min(0,dx)];bb=sk[max(0,-dy):H-max(0,dy),max(0,-dx):W-max(0,dx)];length+=float((aa&bb).sum())*math.hypot(dx*sx,dy*sy)
  for side in [0,1]:
   x=int(np.quantile(xx,.02 if side==0 else .98));chosen=np.abs(xx-x)<=max(2,int(.015*W));entries.append([x/W,float(np.median(yy[chosen])/H)])
 blocks=[]
 for ti,t in enumerate(terminals):
  assigned=[]
  for p in ports:
   if min(range(len(terminals)),key=lambda j:np.linalg.norm(np.array(p['center'])-terminals[j]['center']))==ti:assigned.append(p)
  assigned.sort(key=lambda o:o['center'][1]);entry=entries[min(ti,1)] if entries else None;row=None
  if assigned and entry is not None:row=int(np.argmin([abs(p['center'][1]-entry[1]) for p in assigned]))
  blocks.append({'ports':assigned,'occupied_row':row,'entry':entry,'terminal':t})
 return {'blocks':blocks,'cable_color':color,'cable_length':length}

def extract(cat,image,objects,masks,ocr=None,aux=None):
 if cat in ['breakfast_box','juice_bottle','pushpins']:
  f=loco_extract(cat,image,objects,masks,aux)
 else:
  oo=[{**o,'measured_axis_length':float(np.linalg.norm((np.array(o['axis'][1])-o['axis'][0])*np.array(image.size))/math.hypot(*image.size))} for o in objects]
  f=facts_for(cat,oo,masks,np.array(image.resize((512,512),Image.Resampling.LANCZOS)))
 if cat in VID_LABELS or cat=='engine_wiring':
  if cat=='Ropes':
   f['attributes']['label_text']=ocr;enrich_color(f,masks,np.array(image.resize((512,512),Image.Resampling.LANCZOS)))
  if cat=='engine_wiring':
   g,u=contact(f,masks);f['relation_measurements']=g;f['relation_unknown']=u
  g,u=geometry(cat,f);f['relation_features']=g;f['unknown']=u
 if cat=='screw_bag':
  lengths=sorted(float(np.linalg.norm((np.array(o['axis'][1])-o['axis'][0])*np.array(image.size))) for o in objects if o['class']==2);f['screw_lengths']=lengths
  if len(lengths)==2:f['measurements'].update(short_screw_length=lengths[0],long_screw_length=lengths[1])
 if cat=='splicing_connectors':f.update(connector_facts(image,objects,masks))
 return f

def raw(cat,f):
 """Return named violations and numeric facts; the shared executor fits scales."""
 residual={};unknown=list(f.get('unknown',[]));measures=dict(f['measurements']);counts=f['counts'];attrs=f.get('attributes',{})
 if cat in EXPECTED:
  residual.update({label+'_count':float(abs(counts.get(label,0)-n)) for label,n in zip(CATEGORIES[cat],EXPECTED[cat])})
 if cat in ['breakfast_box','juice_bottle','pushpins']:
  expected={'breakfast_box':{'tangerine':2,'peach':1},'juice_bottle':{'bottle':1,'juice':1,'image_label':1,'text_label':1},'pushpins':{'pushpin':15,'cell':15}}[cat];residual.update({k+'_count':float(abs(counts[k]-n)) for k,n in expected.items()})
  if cat=='breakfast_box':
   residual.update({k+'_presence':float(counts[k]==0) for k in ['oatmeal','dried_fruits','nut','banana_chips']})
  if cat=='juice_bottle':
   residual['fruit_icon_count']=float(abs(f['fruit_icon_count']-1));a,b=f.get('liquid_type'),f.get('fruit_icon_type')
   if a is None or b is None:unknown.append('juice_icon_correspondence')
   else:residual['juice_icon_correspondence']=float(a!=b)
  if cat=='pushpins':residual.update(one_pin_per_cell=float(max((abs(n-1) for n in f['pins_per_detected_cell']),default=1)),pins_outside_cells=float(f['pins_outside_cells']))
 elif cat=='screw_bag':residual.update({k+'_count':float(abs(counts[k]-2)) for k in CATEGORIES[cat]})
 elif cat=='splicing_connectors':
  residual.update(terminal_count=float(abs(counts['terminal']-2)),cable_count=float(abs(counts['cable']-1)));expected={'yellow':2,'blue':3,'red':5}.get(f['cable_color']);blocks=f['blocks']
  if expected is None:unknown.append('cable_color')
  else:residual['color_port_count']=float(max([abs(len(b['ports'])-expected) for b in blocks],default=expected))
  if len(blocks)==2 and all(b['occupied_row'] is not None for b in blocks):residual['port_correspondence']=float(abs(blocks[0]['occupied_row']-blocks[1]['occupied_row']))
  else:unknown.append('occupied_port')
  facing=[]
  for i,b in enumerate(blocks):
   if b['entry'] is not None and b['ports']:
    delta=b['entry'][0]-np.mean([p['center'][0] for p in b['ports']]);facing.append(max(0.,-delta if i==0 else delta)/.1)
  residual['entry_facing']=max(facing,default=1.);measures={'cable_length_'+str(f['cable_color']):f['cable_length']} if f['cable_length'] is not None else {};return residual,measures,unknown
 def centers(label):return [o['center'] for o in attrs.get(label+'_instances',[])]
 if cat=='Tools':
  for label,a,b in [('bolt',0,1/3),('washer',1/3,2/3),('nut',2/3,1)]:residual[label+'_bin']=float(sum(not a<=x<b for x,y in centers(label)))
 if cat=='Blocks':
  for label,a,b in [('circle_block',0,1/3),('triangle_block',1/3,2/3),('square_block',2/3,1)]:residual[label+'_bin']=float(sum(not a<=y<b for x,y in centers(label)))
 if cat=='Stationery':
  for label,left in [('black_pencil',True),('blue_eraser',True),('red_pencil',False),('red_eraser',False)]:residual[label+'_bin']=float(sum(x>=.5 if left else x<.5 for x,y in centers(label)))
  for e,p in [('blue_eraser','black_pencil'),('red_eraser','red_pencil')]:
   if centers(e) and centers(p):residual[e+'_left_of_pencil']=float(centers(e)[0][0]>=centers(p)[0][0])
 for k,v in f.get('relation_features',{}).items():
  if k.startswith('contact_'):measures[k]=v
  elif k.endswith('_error'):residual[k]=float(max(0.,v))
  else:residual[k]=float(max(0.,-v))
 if cat in VID_LABELS or cat in AUTO_LABELS:
  keep=set(continuous_keys(cat))|{k for k in measures if k.startswith('contact_')};measures={k:v for k,v in measures.items() if k in keep}
 if cat=='screw_bag':measures={k:v for k,v in measures.items() if k in ['short_screw_length','long_screw_length']}
 return residual,{k:float(v) for k,v in measures.items() if v is not None},unknown
