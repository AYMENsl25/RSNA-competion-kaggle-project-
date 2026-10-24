"""Resume a single authorized Kaggle cache download with bounded RAM and HTTP Range."""
from pathlib import Path
import json
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'.tools/kaggle'))
os.environ['KAGGLE_CONFIG_DIR']=str(ROOT.parent/'.kaggle')
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def output_url(api,kernel):
    owner,slug=kernel.split('/')
    with api.build_kaggle_client() as client:
        request=ApiListKernelSessionOutputRequest()
        request.user_name=owner;request.kernel_slug=slug
        result=client.kernels.kernels_api_client.list_kernel_session_output(request)
        item=next(f for f in result.files if f.file_name=='slices.npy')
        return item.url

def transfer(url,path,expected):
    offset=path.stat().st_size if path.exists() else 0
    if offset==expected:return
    if offset>expected:raise ValueError('Partial file exceeds expected size')
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with requests.get(url,headers=headers,stream=True,timeout=(30,90)) as response:
        response.raise_for_status()
        if offset:
            cr=response.headers.get('Content-Range','')
            if response.status_code!=206 or not cr.startswith(f'bytes {offset}-') or not cr.endswith(f'/{expected}'):
                raise ValueError('Server did not honor the requested byte range; partial file preserved')
        elif int(response.headers.get('Content-Length',expected))!=expected:
            raise ValueError('Unexpected remote file length')
        with path.open('ab' if offset else 'wb') as f:
            for chunk in response.iter_content(1024*1024):
                if chunk:f.write(chunk)
    if path.stat().st_size!=expected:raise requests.ConnectionError('Incomplete file')

def main():
    kernel,folder=sys.argv[1:]
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text())
    expected=summary['slices_bytes']+128  # NumPy open_memmap v1 header, checked by final SHA256.
    path=folder/'slices.npy'
    api=KaggleApi();api.authenticate()
    deadline=time.monotonic()+6*3600
    for attempt in range(1,101):
        try:
            print(f'Attempt {attempt}: resume byte {path.stat().st_size if path.exists() else 0} / {expected}',flush=True)
            transfer(output_url(api,kernel),path,expected)
            print('Download length complete; pipeline must now verify SHA256',flush=True)
            return
        except requests.RequestException as exc:
            print(f'Interrupted ({type(exc).__name__}); retaining partial bytes',flush=True)
            if time.monotonic()>deadline:raise RuntimeError('Six-hour download deadline exceeded') from None
            time.sleep(min(attempt*3,30))
    raise RuntimeError('Download retry limit exceeded')

if __name__=='__main__':main()
