"""Portable frozen-fact -> score, threshold decision and execution-certificate replay."""
from pathlib import Path
import argparse,gzip,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_source/binding'))
sys.path.insert(0,str(ROOT/'research_source/explanation'))
from original_executor import score
from engine import explain

def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:yield json.loads(line)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source-id');p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    assert not a.out.exists(),'Choose a new output path.'
    cfg=json.loads((ROOT/'configs/SGDB_V2.json').read_text())
    expected={r['source_id']:r for r in rows(ROOT/'data/predictions.jsonl.gz') if r['method']=='SGDB_V2'}
    events={r['source_id']:r['events'] for p in (ROOT/'data/support_events').glob('*/test.jsonl.gz') for r in rows(p)}
    done=rules=alarms=0;err=0.;examples=[]
    for r in rows(ROOT/'data/final_facts.jsonl.gz'):
        if r['split']!='test' or (a.source_id and a.source_id!=r['source_id']):continue
        cat=r['category'];task=r['task'];c=cfg[cat];target=expected[r['source_id']]
        q=score(cat,r['facts'],c,task);threshold=c['thresholds'][task]['threshold']
        delta=abs(q['score']-target['score']);err=max(err,delta)
        assert delta<2e-12 and threshold==target['threshold']
        assert (q['score']>threshold)==target['predicted_anomaly']
        factual={k:r[k] for k in ['objects','facts','association','native_size']}
        report=explain(cat,task,factual,c,events.get(r['source_id'],[]));assert report['score']==q['score']
        report['source_id']=r['source_id'];report['source_sha256']=r['sha256']
        done+=1;rules+=len(report['rules']);alarms+=report['predicted_anomaly']
        if a.source_id:examples.append(report)
    assert done==(1 if a.source_id else 9031)
    assert not any(k in sys.modules for k in ['torch','transformers','tensorflow'])
    result=examples[0] if a.source_id else {'state':'passed','images':done,'executed_rule_records':rules,'alarms':alarms,'max_score_error':err,'threshold_refits':0,'model_forward_calls':0,'input':'frozen facts; not raw-image inference'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ['state','images','executed_rule_records','alarms','max_score_error']},indent=2))
if __name__=='__main__':main()
