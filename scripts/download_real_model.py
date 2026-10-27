"""Rebuild the bundled real-data checkpoint from pinned, checksum-verified weights."""
from _bootstrap import ROOT  # also puts src/ on sys.path
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import numpy as np
import torch
from satsr.geo_io import _atomic_write

COMMIT='8b301688b870f0eb8a5b93337ca3bc90a28734c4'
WEIGHTS_URL='https://drive.usercontent.google.com/download?id=170o8NNBrrIBJqFdoeYCJoyKHvuub0v2k&export=download&confirm=t'
WEIGHTS_SHA='5dae062c73fa43b63519a43a46821047ae40dd2663a8cf79abab4bf224ad13f9'


def rebuild(output):
    from satsr.runtime_support import configure_cpu_threads
    configure_cpu_threads()
    metadata=json.loads((ROOT/'checkpoints/demo/landcover_real_source.json').read_text())
    # The vendored MIT architecture is already supplied; no remote code execution.
    with tempfile.TemporaryDirectory(prefix='l1bsr-real-model-') as name:
        path=Path(name)/'original.pth'
        urllib.request.urlretrieve(WEIGHTS_URL,path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA:
            raise ValueError('Downloaded real weights failed the pinned SHA-256 check')
        original=torch.load(path,map_location='cpu',weights_only=True)
        state={k.removeprefix('module.'):v for k,v in original['state_dict'].items()
               if k.split('.')[-1] not in ('total_ops','total_params')}
        # Architecture mask is tiny and verified against the downloaded author file.
        import io
        raw=urllib.request.urlopen('https://drive.usercontent.google.com/download?id=1WwE2pIHTb7xGql7xQ9TxeZ1pZmk2JhCl&export=download&confirm=t').read()
        if hashlib.sha256(raw).hexdigest() != '7e842657a192b2c3a62d533076a571bdedfdfdf15a92f5d7dc27be07acc1c201':
            raise ValueError('Downloaded architecture mask failed the pinned SHA-256 check')
        mask=np.load(io.BytesIO(raw),allow_pickle=False)
        checkpoint={k:v for k,v in metadata.items() if k not in ('checkpoint_sha256','code_commit')}
        checkpoint.update(mask=mask.tolist(),state_dict=state)
        staged=Path(name)/'converted.pt';torch.save(checkpoint,staged)
        from satsr.models.real_landcover import RealLandCoverEngine
        RealLandCoverEngine(staged,'cpu')  # Strict full-architecture and tensor check.
        _atomic_write(output,lambda temp:temp.write_bytes(staged.read_bytes()))
    print('Saved verified real-data model:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'checkpoints/demo/landcover_real.pt'))
    rebuild(Path(parser.parse_args().output))
