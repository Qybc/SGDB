from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
m=json.loads((R/'FILES_SHA256.json').read_text())
for name,expected in m.items():
    p=R/name;assert p.is_file(),name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
print('PASS: archived file integrity:',len(m),'files')
