import argparse,json,time
import cv2,numpy as np,torch
from PIL import Image
from common import *
from data import preprocess
from model import Model,decode
from train import objects_for
from features import extract
def run(cat,test=False):
 lock=read(ROOT/('execution_lock.json' if test else 'training_lock.json'));assert all(sha(ROOT/n)==h for n,h in lock['files'].items());folder=ROOT/'readers'/cat;done=folder/('test_facts_complete.json' if test else 'normal_facts_complete.json')
 if done.exists():return
 ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);net=Model(len(CATEGORIES[cat])).cuda();net.load_state_dict(ck['model']);net.eval().requires_grad_(False);settings=ck['dev']['settings'];rows=[r for r in read(ROOT/'manifest.json') if r['category']==cat and (r['split']=='test' if test else r['split'] in ['train','dev','calibration'])];ocr={r['source_id']:r for r in lines(ROOT/('rope_ocr_test.jsonl' if test else 'rope_ocr_normal.jsonl'))} if cat=='Ropes' else {};path=folder/('test_facts.jsonl' if test else 'normal_facts.jsonl');(ROOT/'predicted_masks').mkdir(exist_ok=True);start=time.time()
 with path.open('w') as f,torch.inference_mode():
  for j,r in enumerate(rows):
   assert sha(r['image_path'])==r['sha256'];image=Image.open(r['image_path']).convert('RGB')
   with torch.autocast('cuda',dtype=torch.float16):pr=decode(net(preprocess(np.array(image))[None].cuda()))[0]
   if cat=='Ropes':assert r['source_id'] in ocr and ocr[r['source_id']]['input_sha256']==r['sha256']
   objects=objects_for(pr,settings);t=np.array([s['seg_threshold'] for s in settings])[:,None,None];m=(pr['seg']>t).astype('float32');facts=extract(cat,image,objects,m,ocr.get(r['source_id'],{}).get('label_text'));mask=np.uint8(m.argmax(0)+1);mask[m.max(0)<.5]=0;mp=ROOT/'predicted_masks'/(r['sha256']+'.png');Image.fromarray(mask).resize((256,256),Image.Resampling.NEAREST).save(mp)
   f.write(json.dumps({**r,'facts':facts,'objects':objects,'mask_path':str(mp)},ensure_ascii=False,allow_nan=False)+'\n')
   if (j+1)%50==0:f.flush();print(cat,r['split'],j+1,len(rows),flush=True)
 dump(done,{'state':'complete','images':len(rows),'output_sha256':sha(path),'checkpoint_sha256':sha(folder/'best.pt'),'seconds':time.time()-start})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--category',required=True);ap.add_argument('--test',action='store_true');a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1);run(a.category,a.test)
