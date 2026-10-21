"""Download ESA OpenSR SEN2SRLite (RGBN, 4x) weights for compare_sr_models.py.

Only the two weight files and the model card are downloaded. The model card's
load.py is not needed: models/sen2sr_lite.py builds the network itself.

    python download_sen2sr.py
"""
import hashlib
from pathlib import Path
import urllib.request

from models.sen2sr_lite import DEFAULT_DIR

BASE = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/"
FILES = ("model.safetensor", "hard_constraint.safetensor", "mlm.json")


def main(output=DEFAULT_DIR):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = output/name
        if not target.exists():
            request = urllib.request.Request(BASE+name, headers={"User-Agent": "L1BSR-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            temporary = target.with_suffix(target.suffix+".part")
            temporary.write_bytes(data)
            temporary.replace(target)
        print(f"{name}: {hashlib.sha256(target.read_bytes()).hexdigest()}")
    print(f"SEN2SRLite RGBN x4 ready in {output}")


if __name__ == "__main__":
    main()
