"""Validate notebook schemas, code syntax, and embedded runtime imports offline."""
from pathlib import Path
import os
import tempfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
for path in sorted((ROOT/'notebooks'/'v2').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code': compile(cell.source,f'{path.name}:{i}','exec')
    with tempfile.TemporaryDirectory(prefix='rsna_notebook_') as temp:
        os.environ['RSNA_V2_RUNTIME_ROOT']=temp
        scope={}
        exec(nb.cells[2].source,scope)
        for module in ['preprocessing','training','evaluation','submission']:
            __import__('rsna_v2.'+module)
    print(path.name,'PASS syntax/schema/runtime imports')
os.environ.pop('RSNA_V2_RUNTIME_ROOT',None)
