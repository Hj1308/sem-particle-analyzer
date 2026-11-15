# SEM Particle Analyzer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24261363.svg)](https://doi.org/10.5281/zenodo.24261363)
[![CI](https://github.com/Hj1308/sem-particle-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Hj1308/sem-particle-analyzer/actions/workflows/ci.yml)

Watershed-based segmentation and quantitative measurement of particles in
SEM (Scanning Electron Microscopy) images, with a focus on **separating
touching particles** that defeat simple thresholding.

Built for materials science workflows where ImageJ is too manual and
deep-learning is overkill — give it an SEM micrograph and a `nm/pixel`
calibration, and it returns a CSV of per-particle measurements plus an
annotated image.

---

## Features

- **Robust I/O** — handles 8-bit/16-bit TIFFs (including multi-page),
  PNG, and JPG via `tifffile` and OpenCV.
- **Adaptive contrast** — percentile normalization + CLAHE for SEM
  micrographs with non-uniform illumination.
- **Watershed segmentation** — Otsu binarization → distance transform →
  marker-controlled watershed. Splits touching/agglomerated particles.
- **Auto polarity detection** — works for both bright-particles-on-dark
  and dark-particles-on-bright SEM modes (or force it explicitly).
- **Quantitative output** — area, perimeter, equivalent diameter,
  major/minor axes, aspect ratio, circularity, centroid. All calibrated
  to nm.
- **Border filtering** — particles clipped by the image edge are
  excluded by default (their true area is unknown).
- **Reports** — CSV of measurements + annotated TIFF + size-distribution
  histogram.
- **Both CLI and Python API.**

---

## Installation

```bash
git clone https://github.com/Hj1308/sem-particle-analyzer.git
cd sem-particle-analyzer
pip install -e .            # or: pip install -r requirements.txt
```

`pip install -e .` also installs the `sem-particle-analyzer` command
(same options as `python particle_analyzer.py`).

Tested on Python 3.9+.

---

## Quick start

### CLI

```bash
python particle_analyzer.py path/to/image.tif \
    --pixel-size 2.5 \
    --min-area 50 \
    --output results
```

### Python API

```python
from particle_analyzer import analyze

df = analyze(
    "path/to/image.tif",
    output_dir="results",
    pixel_size_nm=2.5,    # nm per pixel (read from the SEM scale bar)
    min_area_px=50,
)

print(df.describe())
print(f"Mean diameter: {df['eq_diameter_nm'].mean():.1f} nm")
```

### Google Colab

```python
!pip install -q tifffile
!git clone https://github.com/Hj1308/sem-particle-analyzer.git
%cd sem-particle-analyzer

from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

from particle_analyzer import analyze
df = analyze(image_path, pixel_size_nm=2.5)
df.head()
```

---

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `image` | — | Input image path (TIFF/PNG/JPG) |
| `-o`, `--output` | `output` | Output directory |
| `-p`, `--pixel-size` | `1.0` | Calibration in nm/pixel |
| `--min-area` | `30` | Minimum particle area (pixels) |
| `--max-area` | none | Maximum particle area (pixels) |
| `--clahe-clip` | `2.0` | CLAHE clip limit (`0` disables) |
| `--gaussian-kernel` | `5` | Gaussian blur kernel (odd) |
| `--seed-h` | `2.0` | Seed depth (px) for watershed: raise if single particles are over-split, lower if touching particles stay merged |
| `--bright-particles` | auto | Force bright-on-dark polarity |
| `--dark-particles` | auto | Force dark-on-bright polarity |
| `--keep-border` | off | Include particles touching the edge |
| `--journal` | `acs` | Format of the size histogram: `acs`, `rsc`, `elsevier` |
| `--fig-width` | `single` | Column width: `single`, `double` (`onehalf` for Elsevier) |
| `--no-lognormal` | off | Do not overlay the log-normal fit |
| `--crop-top`, `--crop-bottom` | `0` | Pixels to remove (e.g. the SEM info bar) |
| `--save-debug` | off | Also save a 2×2 diagnostic panel |

---

## Output

For each input image the output directory receives:

- **`<name>_annotated.tif`** — original image with green contours and
  cyan particle IDs overlaid.
- **`<name>_measurements.csv`** — one row per particle with the columns
  below.
- **`<name>_size_distribution.pdf` / `.tif` / `.png`** — journal-ready
  histogram of equivalent diameters: sized to the chosen journal's column
  width (ACS 8.46/17.78 cm, RSC 8.3/17.1 cm, Elsevier 9/14/19 cm), 7.5 pt
  text, Freedman–Diaconis bins, optional log-normal fit, and N,
  mean ± SD and d₁₀/d₅₀/d₉₀ on the panel. The PDF keeps the text editable;
  the TIFF is LZW-compressed at the journal's resolution (600 dpi ACS/RSC,
  1000 dpi Elsevier). Check your journal's own guide before submission.
- **`<name>_size_statistics.csv`** — N, mean, SD, 95 % confidence interval
  of the mean, d₁₀, d₅₀, d₉₀, span, and the log-normal geometric mean and
  geometric SD. All statistics are number-weighted; say so in the caption.
  Fewer than 100 particles triggers a warning.

### CSV columns

| Column | Unit | Description |
|--------|------|-------------|
| `particle_id` | — | Integer label, matches the annotated image |
| `area_px`, `area_nm2` | px², nm² | Pixel and calibrated area |
| `perimeter_px`, `perimeter_nm` | px, nm | Crofton perimeter estimate (close to the true outline length) |
| `eq_diameter_px`, `eq_diameter_nm` | px, nm | Diameter of a circle with the same area: `2·√(A/π)` |
| `major_axis_nm`, `minor_axis_nm` | nm | Long/short axes of the minimum-area rotated bounding rectangle |
| `aspect_ratio` | — | `major / minor` (≥ 1; `1` = round) |
| `circularity` | — | `4π·A / P²`: ≈ 1.00 ± 0.02 for disks wider than ~30 px, lower for elongated/rough shapes; below ~20 px pixelation lowers it (≈ 0.9 at 16 px) |
| `centroid_x`, `centroid_y` | px | Centroid in image coordinates |

---

## Algorithm

```
        ┌──────────────────┐
        │  Read image      │  tifffile (TIFF)  ▸  cv2.imread (PNG/JPG)
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Normalize 8-bit │  percentile-clip 1–99 % → uint8
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  CLAHE + blur    │  local contrast + Gaussian denoise
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Otsu threshold  │  polarity from the image frame (background)
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Morph open/close│  remove noise, fill small gaps
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Distance transf. │  Euclidean distance to nearest background
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  h-maxima seeds  │  one seed per local distance peak deeper than h
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Watershed       │  on −distance, restricted to the particle mask
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Filter & measure│  size filter, drop edge-touching, compute metrics
        └──────────────────┘
```

---

## Tuning guide

| Symptom | Try |
|---------|-----|
| Touching particles still merged | Lower `--seed-h` (e.g. `1.0`) |
| One particle split into many fragments | Raise `--seed-h` (e.g. `3–5`) |
| Tiny noise blobs counted as particles | Raise `--min-area` |
| Large agglomerates dominating the histogram | Set `--max-area` |
| Background being detected as particles | Try `--dark-particles` or `--bright-particles` |
| Faint particles missed | Raise `--clahe-clip` (e.g. `3.0–4.0`) |
| Edges look jittery | Raise `--gaussian-kernel` to `7` or `9` |

---

## Validation

Synthetic images with known geometry (`tests/synthetic.py`; checked in CI):

| scene | true | found | d₅₀ true → found | d₁₀ true → found |
|---|---|---|---|---|
| 400 separated disks, log-normal radii 3–40 px, clean | 400 | 400 | 21.91 → 21.79 px | 15.8 → 15.7 px |
| same scene, SEM-like (blur σ = 1 px, ±20 % illumination gradient, noise σ = 10) | 400 | 398 | 21.91 → 22.85 px | 15.8 → 16.2 px |
| same scene, v0.1 segmentation (clean) | 400 | 193 | 21.91 → 26.73 px | 15.8 → 22.6 px |

Isolated disks: equivalent diameter within 0.3 % from 16 to 120 px. On blurred,
noisy images the threshold places edges about half a pixel outward (d₅₀ +4 %
above); this matters for particles only a few pixels wide, so image them at a
magnification that gives at least ~10 px per particle diameter.

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The tests build synthetic images with known particle sizes and check
counts, calibration, filters and outputs. Known accuracy defects are listed
in `CHANGELOG.md` and pinned as strict expected failures until fixed.

---

## Limitations

- **Heavy aggregates** with no visible inter-particle valleys cannot be
  split by distance-transform watershed alone. Severe agglomerates may
  need manual segmentation or a learned model (e.g. Cellpose, StarDist).
- **Non-convex particles** (rods, fibers) are measured but the
  `eq_diameter_nm` is only meaningful as an area-equivalent metric.
- **Pixel-size calibration is the user's responsibility** — read it from
  the SEM scale bar or instrument metadata.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If this tool was useful in academic work, please cite it:

```
Jafari, H. sem-particle-analyzer: watershed segmentation and size analysis
of particles in SEM images (v0.2.0). Zenodo, 2026.
https://doi.org/10.5281/zenodo.24261364
```

To cite whichever version is current, use the concept DOI
[10.5281/zenodo.24261363](https://doi.org/10.5281/zenodo.24261363), which
always resolves to the latest release. GitHub's "Cite this repository" button
reads [`CITATION.cff`](CITATION.cff). Report the version you used in the
Experimental section.
