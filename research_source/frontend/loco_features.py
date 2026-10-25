"""Pixel-derived structured facts. No image anomaly labels are consumed."""
import numpy as np
from PIL import Image
from common import LABELS,OUT,mask_for

def region(mask):
    yy,xx=np.where(mask);H,W=mask.shape
    if len(xx)<3:return {'area':float(len(xx)/(H*W)),'cx':None,'cy':None,'top':None,'bottom':None,'left':None,'right':None,'tilt':None}
    pp=np.stack([xx/W,yy/H],1);mu=pp.mean(0);_,v=np.linalg.eigh((pp-mu).T@(pp-mu));axis=v[:,-1]
    return {'area':float(len(xx)/(H*W)),'cx':float(mu[0]),'cy':float(mu[1]),'top':float(yy.min()/H),'bottom':float((yy.max()+1)/H),'left':float(xx.min()/W),'right':float((xx.max()+1)/W),'tilt':float(abs(axis[1]))}
def liquid_descriptor(rgb,masks,regs):
    H,W=rgb.shape[:2];y,x=np.mgrid[:H,:W];m=masks[1].copy();b=regs[0];label=regs[2]
    if b['top'] is not None:
        h=b['bottom']-b['top'];w=b['right']-b['left'];m &= (x/W>b['left']+.2*w)&(x/W<b['right']-.2*w)&(y/H>b['top']+.20*h)&(y/H<b['top']+.42*h)
    if label['top'] is not None:m &= y/H<label['top']-.01
    m &= ~masks[2]&~masks[3]
    if m.sum()<10:
        m=masks[1]&~masks[2]&~masks[3]
        if m.sum():m &= y/H<np.quantile(y[m]/H,.35)
    if m.sum()<10:return None
    pixels=rgb[m].astype(float)/255;return np.quantile(pixels,[.15,.5,.85],axis=0).ravel().tolist()
def classifier_predict(x,model):
    if x is None or model is None:return None,None
    x=(np.array(x)-model['mean'])/model['scale'];z=np.array(model['coef'])@x+model['intercept'];z-=z.max();p=np.exp(z);p/=p.sum();return model['classes'][int(p.argmax())],p.tolist()
def separator_strength(rgb,objects):
    cells=[o for o in objects if o['class']==1]
    if len(cells)!=15:return None
    H,W=rgb.shape[:2];a=np.mean(rgb.astype(float),axis=2)/255;gy,gx=np.gradient(a);cells=sorted(cells,key=lambda o:o['center'][1]);rows=[sorted(cells[i*5:(i+1)*5],key=lambda o:o['center'][0]) for i in range(3)];values=[]
    for row in rows:
        for left,right in zip(row,row[1:]):
            xx=round((left['box'][2]+right['box'][0])/2*W);y0=round(max(left['box'][1],right['box'][1])*H)+3;y1=round(min(left['box'][3],right['box'][3])*H)-3;patch=np.abs(gx[max(0,y0):min(H,y1),max(0,xx-4):min(W,xx+5)])
            if patch.size:values.append(float(np.quantile(patch.max(axis=1),.5)))
    for upper,lower in zip(rows,rows[1:]):
        for top,bottom in zip(upper,lower):
            yy=round((top['box'][3]+bottom['box'][1])/2*H);x0=round(max(top['box'][0],bottom['box'][0])*W)+3;x1=round(min(top['box'][2],bottom['box'][2])*W)-3;patch=np.abs(gy[max(0,yy-4):min(H,yy+5),max(0,x0):min(W,x1)])
            if patch.size:values.append(float(np.quantile(patch.max(axis=0),.5)))
    return float(min(values)) if values else None
def extract(cat,image,objects,masks,colour_model=None):
    rgb=np.array(image.resize(OUT,Image.Resampling.LANCZOS));binary=np.asarray(masks)>.5;regs=[region(m) for m in binary];labels=LABELS[cat];counts={k:sum(o['class']==i for o in objects) for i,k in enumerate(labels)};measure={};meta={}
    if cat=='breakfast_box':
        for i,k in enumerate(labels):
            for f in ['area','cx','cy']:measure[k+'_'+f]=regs[i][f]
        nut,chip=regs[4]['area'],regs[5]['area'];measure['nut_mixture_fraction']=nut/max(1e-8,nut+chip);measure['nut_visible_count']=counts['nut'];measure['banana_visible_count']=counts['banana_chips']
        fruit=[o for o in objects if o['class'] in [2,3]];measure['fruit_rightmost_center']=max((o['center'][0] for o in fruit),default=None)
    elif cat=='juice_bottle':
        icons=sorted([o for o in objects if o['class']>=4],key=lambda o:-o['score']);chosen=[]
        from common import iou
        for o in icons:
            if not any(iou(o['box'],p['box'])>.5 for p in chosen):chosen.append(o)
        meta['fruit_icon_count']=len(chosen);meta['fruit_icon_type']=labels[chosen[0]['class']] if chosen else None;descriptor=liquid_descriptor(rgb,binary,regs);kind,prob=classifier_predict(descriptor,colour_model);meta.update(liquid_colour_descriptor=descriptor,liquid_type=kind,liquid_type_probabilities=prob)
        b=regs[0];bh=(b['bottom']-b['top']) if b['top'] is not None else None;bw=(b['right']-b['left']) if b['left'] is not None else None
        for i,k in enumerate(labels[:4]):
            if i>0:
                for f in ['area','cx','cy']:measure[k+'_'+f]=regs[i][f]
        measure['fill_height_fraction']=(b['bottom']-regs[1]['top'])/max(.001,bh) if bh is not None and regs[1]['top'] is not None else None
        for i,k in [(2,'image_label'),(3,'text_label')]:
            measure[k+'_relative_y']=(regs[i]['cy']-b['top'])/max(.001,bh) if bh is not None and regs[i]['cy'] is not None else None
            measure[k+'_relative_x']=(regs[i]['cx']-b['left'])/max(.001,bw) if bw is not None and regs[i]['cx'] is not None else None
            measure[k+'_tilt']=regs[i]['tilt']
    else:
        pins=[o for o in objects if o['class']==0];cells=[o for o in objects if o['class']==1];assigned=[0]*len(cells);outside=0
        for p in pins:
            candidates=[]
            for j,c in enumerate(cells):
                a,b,d,e=c['box'];x,y=p['center'];w,h=d-a,e-b
                if a-.03*w<=x<=d+.03*w and b-.03*h<=y<=e+.03*h:candidates.append((((x-c['center'][0])/max(.001,w))**2+((y-c['center'][1])/max(.001,h))**2,j))
            if candidates:assigned[min(candidates)[1]]+=1
            else:outside+=1
        meta.update(pins_per_detected_cell=assigned,pins_outside_cells=outside);measure['separator_min_contrast']=separator_strength(np.array(image.resize((640,448),Image.Resampling.LANCZOS)),objects);measure['pushpin_mask_area']=regs[0]['area']
    return {'counts':counts,'measurements':measure,**meta}
def from_annotation(cat,image,objects):
    masks=np.zeros((len(LABELS[cat]),OUT[1],OUT[0]),bool)
    for o in objects:masks[o['class']]|=mask_for(o).astype(bool)
    # Annotation objects receive confidence only to share deterministic readout.
    pp=[{**o,'center':[(o['box'][0]+o['box'][2])/2,(o['box'][1]+o['box'][3])/2],'score':1.} for o in objects]
    return extract(cat,image,pp,masks)
