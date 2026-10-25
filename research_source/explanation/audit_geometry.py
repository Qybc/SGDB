"""Independent source/geometry checks on serialized explanation records."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math

W=Path(__file__).resolve().parent
R=W/'results'
read=lambda p:json.loads(Path(p).read_text())
counts=collections.Counter();maxerr=0.
def axis_length(o,size):
    a,b=o['axis'];return math.hypot((a[0]-b[0])*size[0],(a[1]-b[1])*size[1])
sources={}
with gzip.open(R/'all_explanations.jsonl.gz','rt') as stream:
    for line in stream:
        report=json.loads(line);path=report['source_record_file']
        if path not in sources:
            with gzip.open(path,'rt') if path.endswith('.gz') else Path(path).open() as f:
                sources[path]={r['source_id']:r for r in map(json.loads,f)}
        rec=sources[path][report['source_id']];size=rec['native_size']
        for node in report['rules']:
            if node['kind']!='bound_geometry':continue
            key=node['key'];obs=node['observed'];objs=node['objects']
            if key in ['short_screw_length','long_screw_length']:
                assert len(objs)==1;value=axis_length(objs[0],size)
            elif key.endswith('_length_ratio'):
                # The recorded graph supplies all contributing bound objects.
                groups=collections.defaultdict(list)
                for o in objs:groups[o['class']].append(axis_length(o,size)/math.hypot(*size))
                if report['category']=='Sticks':value=sum(groups[0])/len(groups[0])/max(1e-6,groups[1][0])
                elif report['category']=='Tapes':value=groups[0][0]/max(1e-6,groups[1][0])
                elif report['category']=='Stationery':
                    a,b=(0,2) if key=='pencil_length_ratio' else (1,3)
                    value=groups[a][0]/max(1e-6,groups[b][0])
                elif report['category']=='Ropes':value=groups[0][0]/max(1e-6,groups[1][0])
                else:raise AssertionError(key)
            elif key=='fruit_rightmost_center':value=max(o['center'][0] for o in objs)
            elif key.endswith(('_cx','_cy')):
                axis=0 if key.endswith('_cx') else 1
                value=sum(o['center'][axis] for o in objs)/len(objs)
            else:raise AssertionError(key)
            assert math.isclose(obs,value,abs_tol=1e-9,rel_tol=1e-12),(report['source_id'],key,obs,value)
            maxerr=max(maxerr,abs(obs-value));counts['bound_geometric_operands']+=1
        counts['records']+=1
assert counts['records']==9031
summary={'status':'passed','counts':dict(counts),'maximum_geometry_reconstruction_error':maxerr,
         'method':'Python math from cited saved axes/centers; no engine import or neural/model calls',
         'scope':'Geometry and scalar source consistency, not independent visual correctness.'}
(R/'independent_geometry_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
