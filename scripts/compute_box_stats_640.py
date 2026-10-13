from pathlib import Path
from statistics import median
from PIL import Image


ROOTS = [
    ("COT-AD", Path(r"D:\Desktop\基于无人机的实时棉花幼苗检测\datasets"), [("images", "labels")]),
    ("Cotton-BNYQS", Path(r"D:\python_pro\yolov11\dataset"), [
        ("images/train_improved", "labels/train_improved"),
        ("images/val", "labels/val"),
        ("images/test", "labels/test"),
    ]),
]


def pct(v, q):
    v = sorted(v)
    x = (len(v) - 1) * q
    a = int(x)
    b = min(a + 1, len(v) - 1)
    if a == b:
        return v[a]
    return v[a] * (b - x) + v[b] * (x - a)


combined_widths, combined_heights, combined_areas = [], [], []
for source, root, pairs in ROOTS:
    widths, heights, areas = [], [], []
    invalid = 0
    seen = set()
    for image_rel, label_rel in pairs:
        for image_path in (root / image_rel).glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or "_aug_" in image_path.stem:
                continue
            if source == "COT-AD" and image_path.name.lower() == "16092023_00000123.jpg":
                continue
            if image_path.stem in seen:
                continue
            seen.add(image_path.stem)
            with Image.open(image_path) as im:
                w, h = im.size
            scale = 640.0 / max(w, h)
            label_path = root / label_rel / f"{image_path.stem}.txt"
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, _, _, bw, bh = parts
                bw = float(bw) * w * scale
                bh = float(bh) * h * scale
                if bw <= 0 or bh <= 0:
                    invalid += 1
                    continue
                widths.append(bw)
                heights.append(bh)
                areas.append(bw * bh)
    print(source, "images", len(seen), "valid_boxes", len(areas), "invalid_boxes", invalid)
    for name, values in (("width", widths), ("height", heights), ("area", areas)):
        print(name, *(f"{pct(values,q):.2f}" for q in (0,.25,.5,.75,1)))
    print("area_lt_32sq", sum(a < 1024 for a in areas), f"{100*sum(a < 1024 for a in areas)/len(areas):.2f}%")
    combined_widths.extend(widths)
    combined_heights.extend(heights)
    combined_areas.extend(areas)

print("Combined", "valid_boxes", len(combined_areas))
for name, values in (("width", combined_widths), ("height", combined_heights), ("area", combined_areas)):
    print(name, *(f"{pct(values,q):.2f}" for q in (0,.25,.5,.75,1)))
print("area_lt_32sq", sum(a < 1024 for a in combined_areas), f"{100*sum(a < 1024 for a in combined_areas)/len(combined_areas):.2f}%")
