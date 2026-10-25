"""SGDB screw-bag raw-image inference using existing frozen checkpoints.

No per-image archive facts, test labels, training annotations or expected outputs
are loaded. Only normal-fitted deployment constants are used by the executor.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys,time,subprocess

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'research_source'
for directory in ['frontend','decoder','binding','initial_axis','readout_v1','explanation','language']:
 sys.path.insert(0,str(SRC/directory))
sys.path.insert(0,str(SRC/'frontend'))

def load_readout():
 s=importlib.util.spec_from_file_location('complete_support_v2',SRC/'readout_v2/experiment.py')
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def dump(p,r):
 Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--image',required=True,type=Path)
 ap.add_argument('--frontend',required=True,type=Path)
 ap.add_argument('--decoder',type=Path,default=ROOT/'checkpoints/semantic_decoder/screw_bag/best.pt')
 ap.add_argument('--out',required=True,type=Path)
 ap.add_argument('--llm-model',type=Path,help='Optional local Qwen3-VL-8B-Instruct; no download')
 ap.add_argument('--language',choices=['en','zh'],default='en')
 a=ap.parse_args()
 if a.out.exists():ap.error('Choose a new output directory; existing runs are preserved.')
 import cv2,numpy as np,torch
 from PIL import Image,ImageDraw,ImageFont
 from torch.nn import functional as F
 from common import CATEGORIES
 from data import preprocess
 from model import Model,decode
 from train import objects_for
 from decoder import FactDecoder,decoded,SIZE
 from features import extract
 from axis_adapter import replace_axes
 from hybrid import bind,facts
 from original_executor import relation
 torch.set_num_threads(2);cv2.setNumThreads(1)
 torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 assert torch.cuda.is_available(),'Use a CUDA environment with the frozen model dependencies.'
 # The container launcher restricts physical GPU2 to this one logical device.
 assert torch.cuda.device_count()==1,'Expose only the intended GPU before inference.'
 inventory=json.loads((ROOT/'audit/model_inventory.json').read_text())
 hashes={'image':sha(a.image),'frontend':sha(a.frontend),'decoder':sha(a.decoder)}
 for role,key in [('visual_frontend','frontend'),('semantic_decoder','decoder')]:
  expected=next(r['sha256'] for r in inventory if r['role']==role and r['category']=='screw_bag')
  assert hashes[key]==expected,(role,'checkpoint digest mismatch')
 a.out.mkdir(parents=True)
 started=time.monotonic();timing={}
 im=Image.open(a.image).convert('RGB');rgb=np.array(im);size=np.array(im.size)
 ck=torch.load(a.frontend,map_location='cpu',weights_only=True)
 settings=ck['dev']['settings'];net=Model(3).cuda().eval().requires_grad_(False)
 net.load_state_dict(ck['model']);del ck
 dc=torch.load(a.decoder,map_location='cpu',weights_only=True)
 head=FactDecoder(3,'evidence_edit').cuda().eval().requires_grad_(False)
 head.load_state_dict(dc['model']);del dc
 torch.cuda.synchronize();tick=time.monotonic()
 with torch.inference_mode():
  with torch.autocast('cuda',dtype=torch.float16):out=net(preprocess(rgb)[None].cuda())
  frontend=decode(out)[0];anchors=objects_for(frontend,settings)
  features=F.adaptive_avg_pool2d(net.pixels.float(),SIZE)[0].cpu().half().clone()
  # Preserve the half-precision sigmoid convention used by initial-axis readout.
  axis_mask=out['seg'][0,2].sigmoid().cpu().half().float().numpy()>settings[2]['seg_threshold']
  parent={'features':features,'anchor':anchors,'native_size':list(im.size)}
  d=head([parent]);semantic,slots=decoded(d,0)
  support=d['masks'][0,slots].sigmoid().cpu().numpy()>.5
  torch.cuda.synchronize();timing['visual_and_semantic_forward_seconds']=time.monotonic()-tick
  assert all(not p.requires_grad and p.grad is None for m in [net,head] for p in m.parameters())
 selection=json.loads((SRC/'initial_axis/normal_selection.json').read_text())['chosen']
 witnesses,initial_changes=replace_axes(anchors,axis_mask,im.size,selection)
 mask_threshold=np.array([s['seg_threshold'] for s in settings])[:,None,None]
 masks=(frontend['seg']>mask_threshold).astype('float32')
 fresh_facts=extract('screw_bag',im,anchors,masks)
 scheme=json.loads((ROOT/'configs/initial_selection.json').read_text())['candidate']
 objects,association=bind(semantic,witnesses,scheme)
 bound_facts=facts('screw_bag',fresh_facts,objects,CATEGORIES['screw_bag'],size,relation)
 record={'source_id':'input_'+hashes['image'][:12],'sha256':hashes['image'],
         'split':'inference','task':'screw_bag','category':'screw_bag','dataset':'user_input',
         'native_size':list(im.size),'objects':objects,'facts':bound_facts,'association':association}
 readout=load_readout()
 candidates=readout.v1.candidate_record('screw_bag',record,{2:axis_mask})
 events=readout.extra_checks('screw_bag',record,{2:axis_mask},candidates)
 final=readout.transform('screw_bag',record,events,'complete_support')
 dump(a.out/'facts.json',final)
 dump(a.out/'support_events.json',events)
 subprocess.run([sys.executable,str(ROOT/'scripts/trace_image.py'),
   '--facts',str(a.out/'facts.json'),'--events',str(a.out/'support_events.json'),
   '--out',str(a.out/'execution_trace.json')],check=True)
 report=json.loads((a.out/'execution_trace.json').read_text())
 dump(a.out/'stages.json',{'frontend_objects':anchors,'semantic_objects':semantic,
      'slot_indices':slots,'geometric_witnesses':witnesses,'initial_axis_changes':initial_changes,
      'association':association,'support_events':events,'changed_axes':final['changed_axes']})
 Image.fromarray(np.uint8(axis_mask)*255).save(a.out/'screw_support.png')
 np.savez_compressed(a.out/'decoder_support.npz',support=support)
 im.save(a.out/'input.png')
 canvas=im.copy();draw=ImageDraw.Draw(canvas)
 try:font=ImageFont.truetype('DejaVuSans.ttf',max(16,im.width//55))
 except OSError:font=ImageFont.load_default()
 palette=[(30,135,119),(53,109,178),(196,116,41)]
 for i,o in enumerate(final['objects']):
  if o['box'] is None:continue
  color=palette[o['class']];b=[round(o['box'][k]*size[k%2]) for k in range(4)]
  draw.rectangle(b,outline=color,width=max(2,im.width//400))
  draw.text((b[0]+3,b[1]+3),str(i)+': '+CATEGORIES['screw_bag'][o['class']],fill=color,font=font)
  if o['axis'] is not None:
   ends=(np.array(o['axis'])*size).reshape(-1).tolist();draw.line(ends,fill=color,width=max(3,im.width//300))
 canvas.save(a.out/'evidence_overlay.png')
 del net,head,d,out,parent,features;torch.cuda.empty_cache()
 language=None
 if a.llm_model:
  from qwen_adapter import FrozenQwenGenerator
  from pipeline import explain as realize
  tick=time.monotonic();generator=FrozenQwenGenerator(a.llm_model)
  language=realize(report,generator,a.language)
  timing['llm_load_and_generation_seconds']=time.monotonic()-tick
  dump(a.out/'language.json',language)
  if not language['validation']['valid']:raise RuntimeError('LLM contract rejected; inspect language.json. Detection is unchanged.')
  (a.out/'explanation.txt').write_text(language['text']+'\n')
 timing['total_seconds']=time.monotonic()-started
 result={'category':'screw_bag','predicted_anomaly':report['predicted_anomaly'],
   'decision':'anomaly' if report['predicted_anomaly'] else 'no_alarm',
   'score':report['score'],'threshold':report['threshold'],
   'explanation':language['text'] if language else None,
   'llm_contract_valid':language['validation']['valid'] if language else None,
   'image_sha256':hashes['image'],'evidence_overlay':'evidence_overlay.png','trace':'execution_trace.json'}
 dump(a.out/'result.json',result)
 dump(a.out/'run_receipt.json',{'state':'complete','input':'raw RGB image',
   'hashes':hashes,'torch':torch.__version__,'cuda':torch.version.cuda,
   'device':torch.cuda.get_device_name(0),'visible_device_count':torch.cuda.device_count(),
   'training_steps':0,'baseline_runs':0,'threshold_refits':0,
   'frontend_forward_calls':1,'semantic_decoder_forward_calls':1,
   'per_image_archive_facts_read':False,'timing':timing,
   'configs_sha256':{n:sha(ROOT/'configs'/n) for n in ['SGDB_V2.json','initial_selection.json','quality_floor.json']},
   'llm_model':a.llm_model.name if a.llm_model else None})
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
