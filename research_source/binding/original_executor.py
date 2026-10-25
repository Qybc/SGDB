"""Load unchanged pure scalar functions without importing image/GPU dependencies."""
import ast,copy,math,runpy
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[1]/'frontend'
ns=runpy.run_path(str(SOURCE/'common.py'));ns.update(np=np,math=math,copy=copy)
for filename,names in [('legacy_features.py',['continuous_keys']),('loco_features.py',['classifier_predict']),('features.py',['raw']),('program.py',['apply_aux','residual','score']),('relation_features.py',['geometry'])]:
    tree=ast.parse((SOURCE/filename).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/filename),'exec'),ns)
LABELS=ns['CATEGORIES'];raw=ns['raw'];residual=ns['residual'];score=ns['score'];apply_aux=ns['apply_aux'];relation=ns['geometry']
