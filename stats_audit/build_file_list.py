"""Rebuild the held-out image list exactly as the notebooks do, from the names inside the ImageNetV2 archive.

Notebooks: FILES = sorted(glob(DATA_DIR/"imagenetv2-*"/"*"/"*.jpeg")); perm = RandomState(0).permutation(len(FILES));
build = perm[:500]; held-out = perm[500:1000]. Only file names are read from the archive (streamed, nothing extracted; compression auto-detected as in the notebooks).
"""
import tarfile, urllib.request, json, time, sys
import numpy as np

URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
t0 = time.time()
names = []
with urllib.request.urlopen(URL) as r, tarfile.open(fileobj=r, mode="r|*") as tf:
    for m in tf:
        if m.isfile() and m.name.endswith(".jpeg"):
            names.append(m.name)
print(f"{len(names)} jpeg names read in {time.time()-t0:.0f}s", file=sys.stderr)

# glob pattern imagenetv2-*/*/*.jpeg relative to DATA_DIR; the common "/content/data/" prefix does not change the sort order
files = sorted(n for n in names if n.startswith("imagenetv2-") and n.count("/") == 2)
assert len(files) == 10000, len(files)
perm = np.random.RandomState(0).permutation(len(files))
held = [files[i] for i in perm[500:1000]]
build = [files[i] for i in perm[:500]]
out = {"source_url": URL, "n_files": len(files), "rule": "sorted(glob) then RandomState(0).permutation",
       "held_out": [{"pos": k, "file": f, "class_idx": int(f.split("/")[1])} for k, f in enumerate(held)],
       "build": [{"pos": k, "file": f, "class_idx": int(f.split("/")[1])} for k, f in enumerate(build)]}
json.dump(out, open("data/heldout_files.json", "w"), indent=1)
print("saved data/heldout_files.json; first held-out:", held[:3])
