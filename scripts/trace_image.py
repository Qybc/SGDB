"""CPU rule-execution worker, isolated from neural inference and language models."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_source/explanation'))
from engine import explain

def main():
 p=argparse.ArgumentParser();p.add_argument('--facts',type=Path,required=True)
 p.add_argument('--events',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();assert not a.out.exists()
 r=json.loads(a.facts.read_text());events=json.loads(a.events.read_text())
 cfg=json.loads((ROOT/'configs/SGDB_V2.json').read_text())['screw_bag']
 report=explain('screw_bag','screw_bag',{k:r[k] for k in ['objects','facts','association','native_size']},cfg,events)
 report.update(source_id=r['source_id'],source_sha256=r['sha256'])
 a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 assert not any(m in sys.modules for m in ['torch','transformers','tensorflow'])
if __name__=='__main__':main()
