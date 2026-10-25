import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
V1=Path(os.environ.get('SGDB_FRONTEND_ROOT', str(ROOT.parent/'frontend')))
ZL1=Path(os.environ.get('SGDB_INITIAL_AXIS_ROOT', str(ROOT.parent/'initial_axis')))
VARIANTS=['plain_set','evidence_edit']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=Path(str(p)+'.tmp.'+str(os.getpid()))
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def save(p,x):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=Path(str(p)+'.tmp.'+str(os.getpid()));torch.save(x,q);q.replace(p)
def rows(p):
    with Path(p).open() as f:return [json.loads(l) for l in f]
def verify():
    lock=read(ROOT/'training_lock.json')
    for n,h in lock['files'].items():assert sha(ROOT/n)==h,('source changed',n)
    for n,h in lock['external'].items():assert sha(n)==h,('external changed',n)
    return lock
