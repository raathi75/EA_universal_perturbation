#!/bin/bash
# Downloads the data of the mini-VLM experiment on the LOCAL disk of this machine (/Data is not shared):
#   - ImageNetV2 matched-frequency (~1.2 GB): attack and evaluation images, same as the SmolVLM notebooks
#   - COCO 2017 train + val images and captions (~20 GB): training of the connector, caption quality
# Already downloaded parts are skipped.
set -eo pipefail
D=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
mkdir -p "$D" && cd "$D"

if [ ! -f imagenetv2/.done ]; then
    echo "=== ImageNetV2 ($(date)) ==="
    mkdir -p imagenetv2
    A=imagenetv2-matched-frequency.tar.gz
    # A failed download (HTML error page, a few kB) is removed and downloaded again; the full archive is ~1.2 GB
    if [ -f "$A" ] && [ "$(stat -c %s "$A")" -lt 1000000000 ]; then
        echo "$A is only $(stat -c %s "$A") bytes ($(file -b "$A" | cut -c1-60)): downloading it again"; rm -f "$A"
    fi
    [ -f "$A" ] || wget -q --show-progress -O "$A" "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/$A"
    echo "Archive: $(file -b "$A" | cut -c1-60)"
    # As in the SmolVLM notebooks: the archive may be a tar (compressed or not) or a zip despite its name
    python - "$A" imagenetv2 <<'PY' || exit 1
import sys, tarfile, zipfile
a, d = sys.argv[1], sys.argv[2]
if tarfile.is_tarfile(a):
    with tarfile.open(a, 'r:*') as t:
        t.extractall(d)
elif zipfile.is_zipfile(a):
    with zipfile.ZipFile(a) as z:
        z.extractall(d)
else:
    sys.exit('%s is neither a tar nor a zip archive (failed download?)' % a)
PY
    rm -f "$A" imagenetv2.tar.gz
    n=$(find imagenetv2 -name '*.jpeg' | wc -l)
    [ "$n" = 10000 ] || { echo "ImageNetV2: $n images instead of 10000"; exit 1; }
    touch imagenetv2/.done
fi

mkdir -p coco && cd coco
for z in val2017 train2017; do
    if [ ! -f "$z/.done" ]; then
        echo "=== COCO $z ($(date)) ==="
        wget -c -q --show-progress "http://images.cocodataset.org/zips/$z.zip"
        unzip -q "$z.zip" && rm "$z.zip" && touch "$z/.done"
    fi
done
if [ ! -f annotations/.done ]; then
    wget -c -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip && touch annotations/.done
fi
echo "=== Data ready in $D ($(du -sh "$D" | cut -f1)) ==="
