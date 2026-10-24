import subprocess,time,json,hashlib
from pathlib import Path
out=Path('V2_new_research/outputs/f7_preview_v1'); out.mkdir(parents=True,exist_ok=True)
base=['kaggle','kernels']
for i in range(80):
 r=subprocess.run(base+['status','aymenslimani/rsna-knee-f7-squared-rank'],capture_output=True,text=True)
 if 'COMPLETE' in r.stdout: break
 if any(x in r.stdout for x in ['ERROR','CANCEL']): raise RuntimeError(r.stdout)
 time.sleep(20)
else: raise RuntimeError('Preview polling deadline expired; not submitting')
r=subprocess.run(base+['output','aymenslimani/rsna-knee-f7-squared-rank','-p',str(out)],capture_output=True,text=True)
if r.returncode: raise RuntimeError(r.stderr)
r=json.loads((out/'f7_receipt.json').read_text())
assert r['reader_complete'] and r['rows']==3
assert hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()==r['submission_sha256']
c=subprocess.run(['kaggle','competitions','submit','rsna-knee-abnormality-detection','-k','aymenslimani/rsna-knee-f7-squared-rank','-v','1','-f','submission.csv','-m','F7: fixed squared percentile rank main aggregation; original F0 models coefficients TTA and reader'],capture_output=True,text=True)
(out/'submit_command_result.json').write_text(json.dumps({'returncode':c.returncode,'stdout':c.stdout,'stderr':c.stderr},indent=2))
print(c.stdout,c.stderr,flush=True)
subprocess.run(['kaggle','competitions','submissions','rsna-knee-abnormality-detection','--csv'],stdout=(out/'submissions_after.csv').open('w'))
