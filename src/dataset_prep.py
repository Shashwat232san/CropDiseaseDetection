# src/dataset_prep.py
import os, random, shutil
from pathlib import Path

random.seed(42)
ROOT = Path(r"C:\Projects\CropDiseaseDetection")
RAW = ROOT / "data" / "raw" / "PlantVillage"
OUT = ROOT / "data" / "processed"
splits = {"train":0.7, "val":0.15, "test":0.15}

for cls in [d for d in RAW.iterdir() if d.is_dir()]:
    imgs = list(cls.glob("*.*"))
    random.shuffle(imgs)
    n = len(imgs)
    i1 = int(n * splits["train"])
    i2 = i1 + int(n * splits["val"])
    groups = {"train": imgs[:i1], "val": imgs[i1:i2], "test": imgs[i2:]}
    for split_name, files in groups.items():
        out_dir = OUT / split_name / cls.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(str(f), str(out_dir / f.name))

print("Done splitting dataset.")
