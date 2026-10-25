from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parents[1]
items=json.loads((R/'audit/model_inventory.json').read_text());missing=[];mismatch=[];ok=[]
for r in items:
    p=R/r['path'] if r['included'] else R/'checkpoints/visual_frontend'/r['category']/'best.pt'
    if not p.exists():missing.append({'role':r['role'],'category':r['category'],'expected_sha256':r['sha256']});continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    (ok if h==r['sha256'] else mismatch).append(str(p.relative_to(R)))
print(json.dumps({'verified':len(ok),'missing':missing,'mismatched':mismatch,'raw_image_inference_verified':False},indent=2))
sys.exit(1 if missing or mismatch else 0)
