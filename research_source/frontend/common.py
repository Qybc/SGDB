"""Shared definitions for the single frozen FMP-v1 development rerun."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path(os.environ.get('SGDB_ARCHIVE_ROOT', str(ROOT.parents[1]/'archival_inputs')))
LOCO=BASE/'loco_full_fact_20261018_r0'
R0=BASE/'cross_scene_fact_20261018_r0'
R3=BASE/'relation_fact_refine_20261018_r3'
OLD=BASE/'full_fact_program_20261018_r0'
SCREW=BASE/'screw_frontend_20261018_r1'
VID_LABELS={'Sticks':['blue_stick','red_stick'],'Fruits':['orange','kiwifruit'],'Tools':['bolt','washer','nut'],'Cookies':['yellow_cookie','black_cookie','square_dish','round_dish'],'Tapes':['green_tape','red_tape'],'Stationery':['black_pencil','blue_eraser','red_pencil','red_eraser'],'Ropes':['rope','reference_stick','text_label'],'Blocks':['circle_block','triangle_block','square_block'],'Dishes':['fork','plate','spoon'],'Balls':['orange_ball','white_ball','transparent_case']}
LOCO_LABELS={'breakfast_box':['oatmeal','dried_fruits','tangerine','peach','nut','banana_chips'],'juice_bottle':['bottle','juice','image_label','text_label','orange','banana','cherry'],'pushpins':['pushpin','cell'],'splicing_connectors':['terminal','cable','connector'],'screw_bag':['nut','washer','screw']}
AUTO_LABELS={'engine_wiring':['orange_wire','green_wire','yellow_wire','blue_hoop','metal_clip']}
CATEGORIES={**LOCO_LABELS,**VID_LABELS,**AUTO_LABELS}
LABELS=CATEGORIES
EXPECTED={'Sticks':[2,1],'Fruits':[3,2],'Tools':[2,2,2],'Cookies':[2,1,1,1],'Tapes':[1,1],'Stationery':[1,1,1,1],'Ropes':[1,1,1],'Blocks':[2,2,2],'Dishes':[1,1,1],'Balls':[2,2,1],'engine_wiring':[1,1,1,1,2],'pipe_staple':[1],'tank_screw':[1],'underbody_screw':[1]}
SIZE=(448,320)
OUT=(224,160)
WEIGHTS=Path(os.environ.get('SGDB_IMAGENET_WEIGHTS', str(Path.home()/'.cache/torch/hub/checkpoints/wide_resnet50_2-9ba9bcbe.pth')))
def dump(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def iou(a,b):
 z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return z/max(1e-10,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z)
def mask_for(o,size=OUT):
 import numpy as np
 from PIL import Image,ImageDraw
 im=Image.new('L',size);ImageDraw.Draw(im).polygon([(x*size[0],y*size[1]) for x,y in o['polygon']],fill=1);return np.array(im,dtype='uint8')
