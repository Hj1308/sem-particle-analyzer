"""
SEM Particle Analyzer
=====================
Watershed-based segmentation and measurement of particles in SEM images.

Designed for SEM micrographs where particles are touching or partially
overlapping. Uses Otsu thresholding for binarization, distance transform
for marker generation, and the marker-controlled watershed algorithm for
splitting touching particles.

Outputs per-particle measurements (area, perimeter, equivalent diameter,
circularity, aspect ratio) calibrated to physical units.

Author: Hj1308
License: MIT
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
def read_image(path: str) -> np.ndarray:
    """
    Read an SEM image (TIFF / PNG / JPG).

    Properly handles:
      - 16-bit TIFFs (common for SEM)
      - Multi-page TIFFs (returns the first page)
      - RGB/RGBA images (converted to grayscale)
    """
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        img = tiff.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Could not read image: {path}")

    # Multi-page TIFF -> take first frame
    if img.ndim == 3 and img.shape[0] < 10 and img.shape[-1] not in (3, 4):
        img = img[0]

    # Drop alpha channel
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # To grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


def normalize_to_8bit(img: np.ndarray, low_pct: float = 1.0,
                      high_pct: float = 99.0) -> np.ndarray:
    """Robust percentile-based normalization to uint8."""
    lo, hi = np.percentile(img, (low_pct, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return out.astype(np.uint8)


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def preprocess(gray8: np.ndarray, clahe_clip: float = 2.0,
               clahe_grid: int = 8, gaussian_k: int = 5) -> np.ndarray:
    """CLAHE + Gaussian blur for SEM contrast enhancement."""
    out = gray8.copy()
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip),
                                tileGridSize=(clahe_grid, clahe_grid))
        out = clahe.apply(out)
    if gaussian_k % 2 == 0:
        gaussian_k += 1
    if gaussian_k >= 3:
        out = cv2.GaussianBlur(out, (gaussian_k, gaussian_k), 0)
    return out


# -----------------------------------------------------------------------------
# Segmentation (Otsu + Watershed)
# -----------------------------------------------------------------------------
def segment(gray8: np.ndarray, particles_are_bright: bool | None = None,
            fg_distance_ratio: float = 0.4) -> np.ndarray:
    """
    Otsu threshold + marker-controlled watershed.

    Parameters
    ----------
    gray8 : uint8 grayscale image (preprocessed).
    particles_are_bright : if None, auto-detected from the binary fraction.
    fg_distance_ratio : threshold ratio on the distance transform used to
        seed sure-foreground markers. Increase for tighter splits, decrease
        to merge close peaks (default 0.4).

    Returns
    -------
    labels : int32 array, 0 = background, 1..N = individual particles.
    """
    # Otsu binarization
    _, binary = cv2.threshold(gray8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Auto-detect polarity (bright vs dark particles)
    if particles_are_bright is None:
        white_frac = float(np.sum(binary == 255)) / binary.size
        particles_are_bright = white_frac < 0.5
    if not particles_are_bright:
        binary = cv2.bitwise_not(binary)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance transform -> seeds for watershed
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist, fg_distance_ratio * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Build markers: 1 = background, 2..N = particle seeds, 0 = unknown
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed needs 3-channel input
    bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(bgr, markers)
    # After watershed: -1 = boundary, 1 = background, >=2 = particles

    # Re-pack to 0-based contiguous labels
    labels = np.zeros_like(markers, dtype=np.int32)
    next_id = 1
    for m in np.unique(markers):
        if m <= 1:
            continue
        labels[markers == m] = next_id
        next_id += 1
    return labels


# -----------------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------------
def measure_particles(labels: np.ndarray, pixel_size_nm: float = 1.0,
                      min_area_px: int = 30,
                      max_area_px: int | None = None,
                      touch_border: bool = False) -> pd.DataFrame:
    """
    Compute morphological measurements for every labelled particle.

    Parameters
    ----------
    labels : output of `segment`.
    pixel_size_nm : calibration factor in nanometers per pixel.
    min_area_px / max_area_px : size filters in pixels.
    touch_border : if False (default), particles touching the image edge
        are discarded (their true area is unknown).

    Returns
    -------
    pandas.DataFrame, one row per particle.
    """
    H, W = labels.shape
    rows = []

    for label in range(1, int(labels.max()) + 1):
        mask = (labels == label).astype(np.uint8)
        area_px = int(mask.sum())
        if area_px < min_area_px:
            continue
        if max_area_px is not None and area_px > max_area_px:
            continue

        # Drop edge-touching particles
        if not touch_border:
            ys, xs = np.where(mask)
            if (ys.min() == 0 or xs.min() == 0
                    or ys.max() == H - 1 or xs.max() == W - 1):
                continue

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        perimeter_px = float(cv2.arcLength(cnt, True))
        if perimeter_px <= 0:
            continue

        eq_diam_px = 2.0 * np.sqrt(area_px / np.pi)
        circularity = 4.0 * np.pi * area_px / (perimeter_px ** 2)

        # Rotated bounding rectangle for aspect ratio + axes
        if len(cnt) >= 5:
            (cx, cy), (w, h), _ = cv2.minAreaRect(cnt)
            major = max(w, h)
            minor = min(w, h)
            aspect_ratio = major / minor if minor > 0 else 1.0
        else:
            M = cv2.moments(cnt)
            cx = M["m10"] / M["m00"] if M["m00"] else 0
            cy = M["m01"] / M["m00"] if M["m00"] else 0
            major = minor = 0.0
            aspect_ratio = 1.0

        rows.append({
            "particle_id": label,
            "area_px": area_px,
            "area_nm2": area_px * pixel_size_nm ** 2,
            "perimeter_px": perimeter_px,
            "perimeter_nm": perimeter_px * pixel_size_nm,
            "eq_diameter_px": eq_diam_px,
            "eq_diameter_nm": eq_diam_px * pixel_size_nm,
            "major_axis_nm": major * pixel_size_nm,
            "minor_axis_nm": minor * pixel_size_nm,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "centroid_x": cx,
            "centroid_y": cy,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def annotate(gray8: np.ndarray, labels: np.ndarray, df: pd.DataFrame,
             contour_color=(0, 255, 0),
             label_color=(0, 255, 255)) -> np.ndarray:
    """Overlay particle contours and IDs on the original image."""
    overlay = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    for _, row in df.iterrows():
        pid = int(row["particle_id"])
        mask = (labels == pid).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, contour_color, 1)
        cx, cy = int(row["centroid_x"]), int(row["centroid_y"])
        cv2.putText(overlay, str(pid), (cx - 6, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)
    return overlay


def plot_size_distribution(df: pd.DataFrame, output_path: str,
                           column: str = "eq_diameter_nm",
                           xlabel: str = "Equivalent Diameter (nm)") -> None:
    """Save a histogram of the chosen size column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[column], bins=30, edgecolor="black", alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(
        f"Particle Size Distribution  (N = {len(df)})\n"
        f"mean = {df[column].mean():.1f}   "
        f"median = {df[column].median():.1f}   "
        f"SD = {df[column].std():.1f}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------
def analyze(image_path: str, output_dir: str = "output",
            pixel_size_nm: float = 1.0,
            min_area_px: int = 30, max_area_px: int | None = None,
            clahe_clip: float = 2.0, gaussian_kernel: int = 5,
            fg_distance_ratio: float = 0.4,
            particles_are_bright: bool | None = None,
            keep_border: bool = False,
            verbose: bool = True) -> pd.DataFrame:
    """Run the full pipeline on a single image."""
    os.makedirs(output_dir, exist_ok=True)
    base = Path(image_path).stem

    raw = read_image(image_path)
    if verbose:
        print(f"Read: {image_path}  shape={raw.shape}  dtype={raw.dtype}")

    gray8 = normalize_to_8bit(raw)
    pre = preprocess(gray8, clahe_clip=clahe_clip,
                     gaussian_k=gaussian_kernel)

    labels = segment(pre, particles_are_bright=particles_are_bright,
                     fg_distance_ratio=fg_distance_ratio)
    n_candidates = int(labels.max())
    if verbose:
        print(f"Watershed found {n_candidates} candidate particles.")

    df = measure_particles(labels, pixel_size_nm=pixel_size_nm,
                           min_area_px=min_area_px,
                           max_area_px=max_area_px,
                           touch_border=keep_border)

    if verbose:
        print(f"After filtering: {len(df)} valid particles.")

    if len(df) == 0:
        print("⚠️  No particles remained after filtering. Try adjusting "
              "--min-area, --fg-distance-ratio, or polarity flags.")
        return df

    overlay = annotate(gray8, labels, df)
    annotated_path = os.path.join(output_dir, f"{base}_annotated.tif")
    tiff.imwrite(annotated_path, overlay, photometric="rgb")

    csv_path = os.path.join(output_dir, f"{base}_measurements.csv")
    df.to_csv(csv_path, index=False)

    hist_path = os.path.join(output_dir, f"{base}_size_distribution.png")
    plot_size_distribution(df, hist_path)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"SUMMARY  (N = {len(df)})")
        print(f"{'=' * 50}")
        for col in ["eq_diameter_nm", "circularity", "aspect_ratio"]:
            print(f"{col:>20}:  mean={df[col].mean():>8.2f}   "
                  f"median={df[col].median():>8.2f}   "
                  f"std={df[col].std():>8.2f}")
        print("\n✅  Outputs written to:")
        print(f"   {annotated_path}")
        print(f"   {csv_path}")
        print(f"   {hist_path}")

    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watershed-based particle analysis for SEM images.")
    p.add_argument("image", help="Path to SEM image (TIFF / PNG / JPG)")
    p.add_argument("-o", "--output", default="output",
                   help="Output directory (default: ./output)")
    p.add_argument("-p", "--pixel-size", type=float, default=1.0,
                   help="Calibration in nm/pixel (default: 1.0)")
    p.add_argument("--min-area", type=int, default=30,
                   help="Minimum particle area in pixels (default: 30)")
    p.add_argument("--max-area", type=int, default=None,
                   help="Maximum particle area in pixels (default: none)")
    p.add_argument("--clahe-clip", type=float, default=2.0,
                   help="CLAHE clip limit (0 disables; default: 2.0)")
    p.add_argument("--gaussian-kernel", type=int, default=5,
                   help="Gaussian blur kernel size, odd (default: 5)")
    p.add_argument("--fg-distance-ratio", type=float, default=0.4,
                   help="Distance-transform seed ratio for watershed "
                        "(default: 0.4)")
    polarity = p.add_mutually_exclusive_group()
    polarity.add_argument("--bright-particles", action="store_true",
                          help="Force bright particles on dark background")
    polarity.add_argument("--dark-particles", action="store_true",
                          help="Force dark particles on bright background")
    p.add_argument("--keep-border", action="store_true",
                   help="Keep particles touching the image border")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    if args.bright_particles:
        bright: bool | None = True
    elif args.dark_particles:
        bright = False
    else:
        bright = None
    analyze(
        args.image,
        output_dir=args.output,
        pixel_size_nm=args.pixel_size,
        min_area_px=args.min_area,
        max_area_px=args.max_area,
        clahe_clip=args.clahe_clip,
        gaussian_kernel=args.gaussian_kernel,
        fg_distance_ratio=args.fg_distance_ratio,
        particles_are_bright=bright,
        keep_border=args.keep_border,
    )


if __name__ == "__main__":
    main()
