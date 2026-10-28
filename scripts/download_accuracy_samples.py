"""Download a fixed, evenly spaced eight-image LoveDA validation audit sample.

Academic use only. These are genuine 0.3 m remote-sensing images and labels,
not an exhaustive benchmark. Selection is fixed before inference, never by score.
"""
from _bootstrap import ROOT  # also puts src/ on sys.path
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
from PIL import Image
import numpy as np

REPO = 'chloechia/loveda'
REVISION = '8f77c200865f5f2ab9624bb50bbd06c2034c7411'


def download(root):
    root = Path(root)
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}', timeout=60) as response:
        info = json.load(response)
    files = sorted([f['rfilename'] for f in info['siblings']
                    if f['rfilename'].startswith('urban:rural val images/')], key=lambda p: int(Path(p).stem))
    selected = [files[i] for i in np.linspace(0, len(files)-1, 8, dtype=int)]
    manifest = dict(dataset='LoveDA', split='validation', repository=REPO, revision=REVISION,
        selection='Eight evenly spaced numeric image IDs, selected before inference',
        attribution='Junjue Wang, Zhuo Zheng, Ailong Ma, Xiaoyan Lu, Yanfei Zhong; RSIDEA, Wuhan University',
        license='Academic purposes only; commercial use prohibited; underlying Google Earth terms apply',
        source='https://github.com/Junjue-Wang/LoveDA',
        note='Separate dataset from documented OpenEarthMap training. Small cross-dataset audit; geographic overlap not independently excluded. PNGs have no CRS.', files=[])
    for image in selected:
        for kind, remote in [('images', image), ('labels', image.replace('val images/', 'val masks/'))]:
            relative = f'{kind}/{Path(image).name}'
            dest = root/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'+urllib.parse.quote(remote, safe='/')
            if not dest.exists():
                with urllib.request.urlopen(url, timeout=120) as response:
                    dest.write_bytes(response.read())
            with Image.open(dest) as im:
                im.load()
                if im.size != (1024, 1024):
                    raise ValueError(f'Unexpected LoveDA image dimensions: {relative}')
                if kind == 'labels' and not np.isin(np.asarray(im), range(8)).all():
                    raise ValueError('Unexpected LoveDA class IDs')
            manifest['files'].append(dict(file=relative, url=url, sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        print(Path(image).name, flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    download(ROOT/'tests/fixtures/loveda')
