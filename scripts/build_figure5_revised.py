"""Build the revised Figure 5 for the manuscript.

Figure contract
---------------
Core conclusion: paired ground-truth, prediction, and response-map views make the
qualitative differences between YOLO11n and HCF-YOLO11n auditable without claiming
that detection-derived maps are internal feature activations.
Archetype: image plate + matched local enlargements.
Output: double-column, 183 mm wide; editable labels in SVG/PDF; 600 dpi TIFF.
Evidence: (a,c) full-scene context; (b,d) exactly matched enlargements, with (c,d)
used as a challenging/failure-oriented example rather than a best-case-only panel.
Integrity: crops are identical across columns; no local intensity adjustment; response
maps are existing confidence-weighted box-center maps, each mapped to [0,1] with the
same colormap and display range.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from PIL import Image, ImageDraw
import csv
import cv2
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figure5_revised"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

COLS = [
    ("Ground truth", "ground_truth"),
    ("YOLO11n", "yolov11n_detect.jpg"),
    ("HCF-YOLO11n", "ours_detect.jpg"),
    ("YOLO11n response", "yolov11n_heatmap.jpg"),
    ("HCF-YOLO11n response", "ours_heatmap.jpg"),
]

# Two traceable held-out test scenes. The second is retained as a challenging case.
SCENES = [
    {
        "number": 1,
        "stem": "result_4_97_png.rf.cee5573126d6a64b0cf87daf44934be2",
        "roi": (118, 212, 258, 352),
        "callouts": [
            {"xy": (188, 282), "id": "1", "type": "shared_miss"},
            {"xy": (182, 238), "id": "2", "type": "hcf_stronger"},
            {"xy": (179, 337), "id": "3", "type": "shared_success"},
        ],
        "overview_label": "Dense scene",
        "zoom_label": "Zoom-in",
    },
    {
        "number": 3,
        "stem": "result_6_78_png.rf.7b0f7993ddc2339f888760b9feeb578f",
        "roi": (418, 368, 558, 508),
        "callouts": [
            {"xy": (504, 410), "id": "4", "type": "shared_success"},
            {"xy": (488, 438), "id": "5", "type": "baseline_miss"},
            {"xy": (496, 462), "id": "6", "type": "baseline_miss"},
        ],
        "overview_label": "Challenging scene",
        "zoom_label": "Zoom-in",
    },
]


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def ground_truth(scene: dict) -> Image.Image:
    img = read_rgb(ROOT / f"cotton_scene{scene['number']}" / "original.jpg")
    draw = ImageDraw.Draw(img)
    label_path = ROOT / "dataset" / "labels" / "test" / f"{scene['stem']}.txt"
    w, h = img.size
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        # black under-stroke plus cyan overlay remains legible on soil and foliage
        draw.rectangle((x1, y1, x2, y2), outline=(20, 20, 20), width=2)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 210, 255), width=1)
    return img


def scene_images(scene: dict) -> list[Image.Image]:
    folder = ROOT / f"cotton_scene{scene['number']}"
    images = [
        ground_truth(scene),
        read_rgb(folder / "yolov11n_detect.jpg"),
        read_rgb(folder / "ours_detect.jpg"),
    ]
    images.extend(shared_scale_response_maps(scene))
    return images


def summary_row(stem: str) -> dict[str, str]:
    with open(ROOT / "outputs" / "vis_compare" / "summary.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if Path(row["image_name"]).stem == stem:
                return row
    raise KeyError(stem)


def invert_turbo(path: Path) -> np.ndarray:
    """Recover the stored 0-1 response field from an OpenCV-TURBO JPEG."""
    rgb = np.asarray(read_rgb(path))
    lut_bgr = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)
    lut_rgb = lut_bgr[:, 0, ::-1].astype(np.float32)
    flat = rgb.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    _, indices = cKDTree(lut_rgb).query(unique.astype(np.float32), k=1)
    return indices[inverse].reshape(rgb.shape[:2]).astype(np.float32) / 255.0


def shared_scale_response_maps(scene: dict) -> list[Image.Image]:
    """Apply one per-scene normalization across both model response fields."""
    row = summary_row(scene["stem"])
    name = f"{scene['stem']}.jpg"
    raw_root = ROOT / "outputs" / "vis_compare" / "heatmap_raw"
    baseline = invert_turbo(raw_root / "baseline" / name) * float(row["baseline_max_conf"])
    hcf = invert_turbo(raw_root / "full" / name) * float(row["full_max_conf"])
    shared_max = max(float(baseline.max()), float(hcf.max()), 1e-12)
    rendered = []
    for field in (baseline / shared_max, hcf / shared_max):
        gray = np.clip(np.rint(field * 255), 0, 255).astype(np.uint8)
        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        rendered.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return rendered


def add_callout(ax, scene: dict, callout: dict, mode: str, col: int) -> None:
    gx, gy = callout["xy"]
    x1, y1, _, _ = scene["roi"]
    x, y = (gx, gy) if mode == "overview" else (gx - x1, gy - y1)
    radius = 8.0
    if col == 0:
        ring_color, badge_color, style = "#00A9C6", "#008EA8", "-"
    elif col == 1 and callout["type"] == "shared_success":
        ring_color, badge_color, style = "#E69F00", "#B56F00", "-"
    elif col == 2 and callout["type"] in {"baseline_miss", "shared_success", "hcf_stronger"}:
        ring_color, badge_color, style = "#00A65A", "#008A35", "-"
    elif col in (1, 2) and callout["type"] in {"shared_miss", "baseline_miss"}:
        ring_color, badge_color, style = "#E31A1C", "#D7191C", (0, (2, 1))
    elif col == 1:
        ring_color, badge_color, style = "#E69F00", "#B56F00", "-"
    else:
        ring_color, badge_color, style = "white", "#332A72", "-"
    ax.add_patch(Circle((x, y), radius, fill=False, edgecolor=ring_color, linewidth=1.1, linestyle=style))
    if mode == "zoom":
        # Alternate badge placement to keep nearby examples legible.
        sign = -1 if int(callout["id"]) % 2 else 1
        ax.text(
            x + radius * 0.85,
            y + sign * radius * 0.85,
            callout["id"],
            color="white",
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.14", "facecolor": badge_color, "edgecolor": "white", "linewidth": 0.5},
        )


fig = plt.figure(figsize=(183 / 25.4, 158 / 25.4), facecolor="white")
gs = fig.add_gridspec(
    nrows=4,
    ncols=5,
    left=0.055,
    right=0.995,
    bottom=0.035,
    top=0.955,
    wspace=0.018,
    hspace=0.055,
)

panel_letters = ["a", "b", "c", "d"]
row_idx = 0
for scene in SCENES:
    imgs = scene_images(scene)
    x1, y1, x2, y2 = scene["roi"]
    for mode in ("overview", "zoom"):
        for col, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if mode == "overview":
                shown = img
            else:
                shown = img.crop((x1, y1, x2, y2))
            ax.imshow(shown, interpolation="none")
            if mode == "overview":
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.8,
                        linestyle=(0, (3, 2)),
                    )
                )
            if mode == "zoom":
                for callout in scene["callouts"]:
                    add_callout(ax, scene, callout, mode, col)
            if row_idx == 0:
                ax.set_title(COLS[col][0], fontsize=6.7, pad=2.2, fontweight="bold")

        left_ax = fig.axes[row_idx * 5]
        left_ax.text(
            -0.065,
            1.015,
            panel_letters[row_idx],
            transform=left_ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        left_ax.text(
            -0.065,
            0.5,
            scene["overview_label"] if mode == "overview" else scene["zoom_label"],
            transform=left_ax.transAxes,
            fontsize=6.2,
            rotation=90,
            va="center",
            ha="right",
        )
        row_idx += 1

# Compact method-color key and explicit failure annotations.
fig.text(0.055, 0.015, "GT: cyan", color="#008EA8", fontsize=6.2, weight="bold")
fig.text(0.145, 0.015, "YOLO11n: orange", color="#C26A00", fontsize=6.2, weight="bold")
fig.text(0.292, 0.015, "HCF-YOLO11n: green", color="#008A35", fontsize=6.2, weight="bold")
fig.text(
    0.505,
    0.015,
    "1 shared miss;  2 stronger HCF;  3-4 shared TP;  5-6 recovered by HCF",
    fontsize=6.0,
    fontweight="semibold",
)

base = OUT / "Figure5_revised_zoom_in"
fig.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")
fig.savefig(base.with_suffix(".pdf"), facecolor="white")
fig.savefig(base.with_suffix(".svg"), facecolor="white")
plt.close(fig)

caption = (
    "Figure 5. Qualitative comparison of YOLO11n and HCF-YOLO11n in dense cotton seedling scenes. "
    "(a,c) Full-scene ground-truth annotations, predictions, and detection-derived response maps for a representative "
    "dense scene and a challenging scene, respectively. Dashed white rectangles indicate the regions enlarged in "
    "(b,d). (b,d) Matched local enlargements using identical crop coordinates across all columns. Ground-truth boxes "
    "are cyan, YOLO11n predictions are orange, and HCF-YOLO11n predictions are green. Numbered circles identify "
    "success- and failure-oriented examples: 1, a seedling missed by both models; 2, a case with a stronger HCF-YOLO11n "
    "response; 3-4, seedlings detected by both models; and 5-6, YOLO11n misses recovered by HCF-YOLO11n. "
    "Response maps were generated from confidence-weighted Gaussian kernels centered on predicted boxes "
    "(radius = max(3, 0.5 min(w,h)); sigma = max(radius/2,1)); overlapping responses were combined by the pixelwise "
    "maximum. For each scene, the two response fields were jointly normalized by their shared maximum and rendered "
    "with the same color map and [0,1] display range. No internal network activation layer was used. These "
    "panels are qualitative examples and are interpreted together with the quantitative detection metrics."
)
(OUT / "Figure5_caption.txt").write_text(caption, encoding="utf-8")
print(base)
