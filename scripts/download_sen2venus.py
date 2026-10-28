"""Download and extract SEN2VENµS site archives (Zenodo record 6514159, CC BY-NC 4.0).

Michel, Vinasco-Salinas, Inglada & Hagolle (2022), "SEN2VENµS, a Dataset for the
Training of Sentinel-2 Super-Resolution Algorithms", Data 7(7):96.

Downloads are resumable and MD5-checked. Archives go to datasets/sen2venus/archives,
extracted sites to datasets/sen2venus/<SITE>/ (both gitignored).
"""
import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECORD = "https://zenodo.org/api/records/6514159"
FILE_URL = "https://zenodo.org/records/6514159/files/{name}"
SPLITS = {
    "train": ["ATTO", "ESTUAMAR", "SUDOUE-4", "K34-AMAZ", "MAD-AMBO", "LERIDA-1", "JAM2018",
              "NARYN", "ANJI", "SO2", "FGMANAUS", "ES-LTERA", "SUDOUE-5", "BAMBENW2", "ESGISB-1",
              "SUDOUE-6", "FR-LQ1"],
    "val": ["ALSACE"],
    "test": ["BENGA", "KUDALIAR"],
}
DEFAULT_SITES = [site for sites in SPLITS.values() for site in sites]


def md5sum(path, chunk=8 * 1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(url, part, start, end, progress):
    """Fetch bytes [start, end] into `part`, resuming from what is already there."""
    for attempt in range(20):
        have = part.stat().st_size if part.exists() else 0
        if start + have > end:
            return
        request = urllib.request.Request(url, headers={"Range": f"bytes={start+have}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError("Server does not support range requests")
                with open(part, "ab") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                        progress(len(block))
        except (OSError, urllib.error.URLError) as error:
            print(f"  {part.name}: retry {attempt+1} after {error}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    if part.stat().st_size != end - start + 1:
        raise RuntimeError(f"{part.name}: incomplete after retries; re-run to resume")


def download(url, target, size, connections=1):
    """Resumable download split into `connections` parallel byte ranges.

    Zenodo throttles each connection, so several ranges finish much sooner.
    """
    connections = max(1, min(connections, 16))
    step = -(-size // connections)
    ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(connections)]
    parts = [target.with_name(f"{target.name}.part{i}") for i in range(connections)]
    done = [sum(p.stat().st_size for p in parts if p.exists())]
    lock, last = threading.Lock(), [-1]

    def progress(n):
        with lock:
            done[0] += n
            percent = int(100 * done[0] / size)
            if percent != last[0] and percent % 5 == 0:
                print(f"  {target.name}: {percent}% ({done[0]/1e9:.2f}/{size/1e9:.2f} GB)", flush=True)
                last[0] = percent

    with ThreadPoolExecutor(connections) as pool:
        for future in [pool.submit(fetch_range, url, p, s, e, progress) for p, (s, e) in zip(parts, ranges)]:
            future.result()
    tmp = target.with_name(target.name + ".joining")
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as handle:
                shutil.copyfileobj(handle, out, 16 * 1024 * 1024)
    tmp.replace(target)
    for part in parts:
        part.unlink()


def extract(archive, destination):
    import py7zr
    marker = destination / ".extracted"
    if marker.exists():
        return
    print(f"  extracting {archive.name} ...", flush=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with py7zr.SevenZipFile(archive, "r") as handle:
        handle.extractall(tmp)
    shutil.rmtree(destination, ignore_errors=True)
    tmp.replace(destination)
    marker.write_text("ok", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--root", default="datasets/sen2venus")
    parser.add_argument("--connections", type=int, default=6, help="Parallel byte ranges per file (1-16)")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--keep-archives", action="store_true", help="Keep .7z files after extraction")
    args = parser.parse_args()
    root = Path(args.root)
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD, timeout=60) as response:
        files = {f["key"]: f for f in json.load(response)["files"]}
    missing = [s for s in args.sites if f"{s}.7z" not in files]
    if missing:
        sys.exit(f"Unknown sites: {missing}")
    for site in args.sites:
        meta = files[f"{site}.7z"]
        expected = meta["checksum"].split(":", 1)[1]
        archive = archives / f"{site}.7z"
        if (root / site / ".extracted").exists():
            print(f"{site}: already extracted", flush=True)
            continue
        print(f"{site}: {meta['size']/1e9:.2f} GB", flush=True)
        if not archive.exists():
            download(FILE_URL.format(name=f"{site}.7z"), archive, meta["size"], args.connections)
        actual = md5sum(archive)
        if actual != expected:
            archive.unlink()
            sys.exit(f"{site}: MD5 mismatch ({actual} != {expected}); deleted, re-run to retry")
        print(f"  MD5 ok", flush=True)
        if not args.no_extract:
            extract(archive, root / site)
            if not args.keep_archives:
                archive.unlink()
    print("done", flush=True)


if __name__ == "__main__":
    main()
