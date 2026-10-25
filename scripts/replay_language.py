"""Check recorded LLM text against its saved evidence contract, offline."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_source/language'))
from contracts import validate
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert not a.out.exists()
    contracts=json.loads((ROOT/'language_records/contracts.json').read_text())
    reports=json.loads((ROOT/'language_records/final_reports.json').read_text());checks=[]
    for r in reports:
        c=contracts[r['request_id']];q=validate(r['raw_llm_output'],c)
        assert q['valid'] and q==r['validation'],r['request_id']
        assert r['score']==c['score'] and r['threshold']==c['threshold'] and not r['detector_modified']
        checks.append({'request_id':r['request_id'],'source_id':r['source_id'],'valid':q['valid']})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'state':'passed','records':len(checks),'new_LLM_calls':0,'interpretation':'Mechanical evidence contract replay; not an independent human factuality assessment.','checks':checks},indent=2)+'\n')
    print('Recorded language contracts verified:',len(checks))
if __name__=='__main__':main()
