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
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


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
            h: float = 2.0, fg_distance_ratio: float | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Otsu threshold + marker-controlled watershed on the distance transform.

    Parameters
    ----------
    gray8 : uint8 grayscale image (preprocessed).
    particles_are_bright : None (default) decides the polarity from the image
        frame: the class that occupies most of the outer one-pixel border is
        taken as background. This works when particles cover most of the field.
    h : seed depth in pixels. Each particle is seeded at the regional maxima of
        the Euclidean distance transform that rise more than `h` above their
        surroundings (h-maxima). The criterion is local, so small particles get
        their own seeds next to large ones. Raise `h` if single particles are
        over-split, lower it if touching particles stay merged (default 2.0).
    fg_distance_ratio : deprecated and ignored (the former global seed rule,
        `dist > ratio * dist.max()`, dropped small particles).

    Returns
    -------
    labels : int32 array, 0 = background, 1..N = individual particles. Every
        foreground pixel belongs to a particle (no watershed boundary ring).
    binary : uint8 binary mask used by watershed (for debugging).
    """
    if fg_distance_ratio is not None:
        warnings.warn(
            "fg_distance_ratio is ignored since v0.2: seeds now use h-maxima of the "
            "distance transform; tune `h` instead.", DeprecationWarning, stacklevel=2)
    empty = np.zeros(gray8.shape, np.int32)
    if int(gray8.max()) == int(gray8.min()):  # featureless frame: nothing to segment
        return empty, np.zeros(gray8.shape, np.uint8)

    # Otsu binarization (255 = brighter class)
    _, binary = cv2.threshold(gray8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Polarity: the background dominates the outer frame of the field of view
    if particles_are_bright is None:
        frame = np.concatenate([binary[0, :], binary[-1, :],
                                binary[:, 0], binary[:, -1]])
        particles_are_bright = float(np.mean(frame == 255)) < 0.5
    if not particles_are_bright:
        binary = cv2.bitwise_not(binary)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = binary > 0
    if not mask.any() or mask.all():
        return empty, binary

    # Distance transform -> h-maxima seeds -> watershed restricted to the mask
    dist = ndi.distance_transform_edt(mask)
    markers = cc_label(h_maxima(dist, h))
    # A component with no seed (a particle narrower than ~2h, or h set too high)
    # would vanish; seed it at its own distance maximum instead.
    comps, n_comp = ndi.label(mask)
    seeded = np.zeros(n_comp + 1, bool)
    seeded[np.unique(comps[markers > 0])] = True
    missing = [i for i in range(1, n_comp + 1) if not seeded[i]]
    if missing:
        peaks = ndi.maximum_position(dist, comps, missing)
        nxt = int(markers.max()) + 1
        for k, (r, c) in enumerate(peaks):
            markers[r, c] = nxt + k
    labels = watershed(-dist, markers, mask=mask).astype(np.int32)
    return labels, binary


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

    The perimeter is the Crofton estimate (scikit-image), which is close to the
    true length of a smooth outline; circularity = 4*pi*A/P^2 is then about 1
    for disks wider than ~30 px and drops below ~0.9 for very small particles,
    where pixelation dominates.

    Returns
    -------
    pandas.DataFrame, one row per particle.
    """
    H, W = labels.shape
    rows = []
    for rp in regionprops(labels):
        area_px = int(rp.area)
        if area_px < min_area_px:
            continue
        if max_area_px is not None and area_px > max_area_px:
            continue
        r0, c0, r1, c1 = rp.bbox  # r1, c1 exclusive
        if not touch_border and (r0 == 0 or c0 == 0 or r1 == H or c1 == W):
            continue
        perimeter_px = float(rp.perimeter_crofton)
        if perimeter_px <= 0:
            continue
        eq_diam_px = 2.0 * np.sqrt(area_px / np.pi)
        circularity = 4.0 * np.pi * area_px / (perimeter_px ** 2)

        # Rotated bounding rectangle for aspect ratio + axes
        crop = rp.image.astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            _, (w, h), _ = cv2.minAreaRect(cnt)
            major, minor = max(w, h), min(w, h)
            aspect_ratio = major / minor if minor > 0 else 1.0
        else:
            major = minor = 0.0
            aspect_ratio = 1.0
        cy, cx = rp.centroid

        rows.append({
            "particle_id": int(rp.label),
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
            "centroid_x": float(cx),
            "centroid_y": float(cy),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def annotate(gray8: np.ndarray, labels: np.ndarray, df: pd.DataFrame,
             contour_color=(0, 255, 0),
             label_color=(0, 255, 255),
             thickness: int | None = None,
             font_scale: float | None = None) -> np.ndarray:
    """
    Overlay particle contours and IDs on the original image.

    Contour thickness and font size auto-scale with image dimensions so
    output is visible on both small (512 px) and large (3000+ px) SEM
    micrographs. Pass `thickness` / `font_scale` to override.
    """
    H, W = gray8.shape[:2]
    if thickness is None:
        thickness = max(1, int(round(min(H, W) / 600)))
    if font_scale is None:
        font_scale = max(0.4, min(H, W) / 1500.0)

    overlay = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    for _, row in df.iterrows():
        pid = int(row["particle_id"])
        mask = (labels == pid).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, contour_color, thickness)
        cx, cy = int(row["centroid_x"]), int(row["centroid_y"])
        cv2.putText(overlay, str(pid),
                    (cx - int(6 * font_scale), cy + int(4 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    label_color, max(1, thickness), cv2.LINE_AA)
    return overlay


def save_debug_panel(gray8: np.ndarray, binary: np.ndarray,
                     labels: np.ndarray, output_path: str) -> None:
    """
    Save a 2x2 diagnostic panel:
        [preprocessed]   [Otsu mask]
        [watershed labels]   [contours]
    Useful to see WHY segmentation succeeded or failed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(gray8, cmap="gray")
    axes[0, 0].set_title("1. Preprocessed grayscale")

    pct = int(np.sum(binary == 255) / binary.size * 100)
    axes[0, 1].imshow(binary, cmap="gray")
    axes[0, 1].set_title(f"2. Otsu binary  (white = {pct}% of frame)")

    rng = np.random.default_rng(42)
    color_lut = rng.integers(40, 255,
                             size=(int(labels.max()) + 1, 3),
                             dtype=np.uint8)
    color_lut[0] = [0, 0, 0]
    axes[1, 0].imshow(color_lut[labels])
    axes[1, 0].set_title(f"3. Watershed labels  (N = {int(labels.max())})")

    over = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    H, W = gray8.shape[:2]
    th = max(1, int(round(min(H, W) / 600)))
    for label in range(1, int(labels.max()) + 1):
        m = (labels == label).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, (0, 255, 0), th)
    axes[1, 1].imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("4. Contours overlay")

    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


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
# Size statistics and publication figure
# -----------------------------------------------------------------------------
# Printed sizes from the publishers' artwork guides (checked November 2026):
# ACS single column up to 3.33 in, double 7 in, lettering >= 4.5 pt;
# RSC 8.3 / 17.1 cm, TIFF >= 600 dpi; Elsevier 90 / 140 / 190 mm, text >= 6 pt
# (about 7 pt recommended), 1000 dpi for line art. Same values as the
# chem-figures conventions used across these tools.
JOURNALS = {
    "acs": {"widths_cm": {"single": 8.46, "double": 17.78}, "dpi": 600},
    "rsc": {"widths_cm": {"single": 8.3, "double": 17.1}, "dpi": 600},
    "elsevier": {"widths_cm": {"single": 9.0, "onehalf": 14.0, "double": 19.0}, "dpi": 1000},
}
_BAR_COLOR = "#0072B2"  # Okabe-Ito blue; black edges keep bars distinct in grayscale


def _gum_decimals(u: float) -> int:
    e = int(np.floor(np.log10(u)))
    return (2 if int(u / 10**e + 1e-9) < 3 else 1) - 1 - e


def _round_pm(value: float, u: float) -> tuple[str, str]:
    """Round u to 2 significant digits if it starts with 1 or 2, else 1, and the value
    to the same decimal place (GUM, JCGM 100:2008 7.2.6). Returns two strings."""
    if not np.isfinite(u) or u <= 0:
        return f"{value:.3g}", "0"
    d = _gum_decimals(u)
    if _gum_decimals(round(u, d)) != d:  # rounding carried, e.g. 0.096 -> 0.10
        d = _gum_decimals(round(u, d))
    k = max(d, 0)
    return f"{round(value, d):.{k}f}", f"{round(u, d):.{k}f}"


def size_statistics(values) -> dict:
    """Number-weighted size statistics of positive values (e.g. equivalent diameters).

    Returns N, mean, SD, the 95 % confidence interval of the mean (t-based),
    d10 / d50 / d90 (number percentiles), span = (d90 - d10) / d50, and the
    log-normal maximum-likelihood fit: geometric mean (median of the fitted
    distribution) and geometric standard deviation sigma_g."""
    from scipy import stats as sstats

    x = np.asarray(values, float)
    x = x[np.isfinite(x) & (x > 0)]
    n = int(x.size)
    out = {"N": n}
    if n == 0:
        return out
    d10, d50, d90 = np.percentile(x, [10, 50, 90])
    out.update(mean=float(x.mean()), median=float(d50), d10=float(d10), d50=float(d50),
               d90=float(d90), span=float((d90 - d10) / d50) if d50 > 0 else np.nan)
    if n >= 2:
        sd = float(x.std(ddof=1))
        half = float(sstats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        lx = np.log(x)
        out.update(sd=sd, ci95_mean=(out["mean"] - half, out["mean"] + half),
                   geo_mean=float(np.exp(lx.mean())), geo_sd=float(np.exp(lx.std(ddof=1))))
    return out


def plot_size_distribution_publication(
    df: pd.DataFrame,
    output_base: str,
    column: str = "eq_diameter_nm",
    xlabel: str = "Equivalent diameter (nm)",
    journal: str = "acs",
    width: str = "single",
    lognormal: bool = True,
    unit: str = "nm",
) -> dict:
    """Journal-ready size histogram, saved as PDF (vector, editable text), TIFF and PNG.

    Sized to the journal column width, 7.5 pt text, inward ticks, Freedman-Diaconis
    bins, counts on the y axis (bars start at 0), an optional log-normal fit drawn
    with the same count scaling, and the statistics printed on the panel
    (N, mean +/- SD, d10-d50-d90). Returns the statistics dict plus 'files' and
    'warnings'. State in the caption that the distribution is number-weighted."""
    if journal not in JOURNALS:
        raise ValueError(f"journal must be one of {sorted(JOURNALS)}")
    spec = JOURNALS[journal]
    if width not in spec["widths_cm"]:
        raise ValueError(f"width must be one of {sorted(spec['widths_cm'])} for {journal}")
    x = df[column].to_numpy(float)
    x = x[np.isfinite(x) & (x > 0)]
    st = size_statistics(x)
    warns = []
    if st["N"] < 100:
        warns.append(f"Only {st['N']} particles: percentiles and the fit are uncertain; "
                     "measure more particles (several images) if possible.")
    w_in = spec["widths_cm"][width] / 2.54
    rc = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    files = {}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(w_in, w_in * 0.72))
        if st["N"] > 0:
            counts, edges, _ = ax.hist(x, bins="fd" if st["N"] >= 4 else "auto",
                                       color=_BAR_COLOR, edgecolor="black", linewidth=0.5)
            if lognormal and st["N"] >= 10 and st.get("geo_sd", 1) > 1:
                from scipy import stats as sstats
                xs = np.linspace(max(edges[0], 1e-9), edges[-1], 400)
                pdf = sstats.lognorm.pdf(xs, s=np.log(st["geo_sd"]), scale=st["geo_mean"])
                ax.plot(xs, pdf * st["N"] * (edges[1] - edges[0]), color="black", lw=1.0,
                        label="Log-normal fit")
                ax.legend(loc="upper right", handlelength=1.5)
            m, sd = _round_pm(st["mean"], st.get("sd", np.nan))
            text = [f"N = {st['N']}", f"mean ± SD = {m} ± {sd} {unit}",
                    f"d$_{{10}}$/d$_{{50}}$/d$_{{90}}$ = {st['d10']:.3g}/{st['d50']:.3g}/"
                    f"{st['d90']:.3g} {unit}"]
            ax.text(0.98, 0.78 if lognormal else 0.97, "\n".join(text), transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5)
            ax.set_ylim(0, max(counts.max() * 1.45, 1))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        fig.tight_layout(pad=0.3)
        os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
        for ext in ("pdf", "tif", "png"):
            path = f"{output_base}.{ext}"
            kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
            fig.savefig(path, dpi=spec["dpi"], **kw)
            files[ext] = path
        plt.close(fig)
    st.update(files=files, warnings=warns, journal=journal, width=width)
    return st


# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------
def analyze(image_path: str, output_dir: str = "output",
            pixel_size_nm: float = 1.0,
            min_area_px: int = 30, max_area_px: int | None = None,
            clahe_clip: float = 2.0, gaussian_kernel: int = 5,
            seed_h: float = 2.0,
            fg_distance_ratio: float | None = None,
            particles_are_bright: bool | None = None,
            keep_border: bool = False,
            crop_top_px: int = 0,
            crop_bottom_px: int = 0,
            save_debug: bool = False,
            journal: str = "acs",
            fig_width: str = "single",
            lognormal_fit: bool = True,
            verbose: bool = True) -> pd.DataFrame:
    """Run the full pipeline on a single image.

    Extra parameters
    ----------------
    crop_top_px, crop_bottom_px : pixels to remove from the top/bottom
        before processing. Use this to strip the SEM info bar (commonly
        60-150 px tall at the bottom).
    save_debug : if True, also saves a 2x2 diagnostic panel showing the
        intermediate stages (preprocessed / binary / labels / contours).
    journal, fig_width : target journal ('acs', 'rsc', 'elsevier') and column
        width ('single', 'double'; Elsevier also 'onehalf') for the
        publication histogram <name>_size_distribution.{pdf,tif,png}.
    lognormal_fit : overlay the log-normal maximum-likelihood fit.
    A <name>_size_statistics.csv with N, mean, SD, 95 % CI of the mean,
    d10/d50/d90, span and the log-normal parameters is also written.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = Path(image_path).stem

    raw = read_image(image_path)
    if verbose:
        print(f"Read: {image_path}  shape={raw.shape}  dtype={raw.dtype}")

    # Strip SEM info bar regions before analysis
    if crop_top_px > 0 or crop_bottom_px > 0:
        H = raw.shape[0]
        bottom = H - crop_bottom_px if crop_bottom_px > 0 else H
        raw = raw[crop_top_px:bottom, :]
        if verbose:
            print(f"Cropped to: {raw.shape}  "
                  f"(removed {crop_top_px}px top, {crop_bottom_px}px bottom)")

    gray8 = normalize_to_8bit(raw)
    pre = preprocess(gray8, clahe_clip=clahe_clip,
                     gaussian_k=gaussian_kernel)

    labels, binary = segment(pre,
                             particles_are_bright=particles_are_bright,
                             h=seed_h, fg_distance_ratio=fg_distance_ratio)
    n_candidates = int(labels.max())
    if verbose:
        print(f"Watershed found {n_candidates} candidate particles.")

    if save_debug:
        debug_path = os.path.join(output_dir, f"{base}_debug.png")
        save_debug_panel(pre, binary, labels, debug_path)
        if verbose:
            print(f"Debug panel saved: {debug_path}")

    df = measure_particles(labels, pixel_size_nm=pixel_size_nm,
                           min_area_px=min_area_px,
                           max_area_px=max_area_px,
                           touch_border=keep_border)

    if verbose:
        print(f"After filtering: {len(df)} valid particles.")

    if len(df) == 0:
        print("⚠️  No particles remained after filtering. Try adjusting "
              "--min-area, --seed-h, polarity flags, or "
              "crop_bottom_px to remove the SEM info bar.")
        return df

    overlay = annotate(gray8, labels, df)
    annotated_path = os.path.join(output_dir, f"{base}_annotated.tif")
    tiff.imwrite(annotated_path, overlay, photometric="rgb")

    csv_path = os.path.join(output_dir, f"{base}_measurements.csv")
    df.to_csv(csv_path, index=False)

    fig_stats = plot_size_distribution_publication(
        df, os.path.join(output_dir, f"{base}_size_distribution"),
        journal=journal, width=fig_width, lognormal=lognormal_fit)
    hist_path = fig_stats["files"]["pdf"]
    stats_row = {k: v for k, v in fig_stats.items()
                 if k not in ("files", "warnings", "ci95_mean")}
    if "ci95_mean" in fig_stats:
        stats_row["ci95_mean_low"], stats_row["ci95_mean_high"] = fig_stats["ci95_mean"]
    stats_path = os.path.join(output_dir, f"{base}_size_statistics.csv")
    pd.DataFrame([stats_row]).to_csv(stats_path, index=False)
    for w in fig_stats["warnings"]:
        print(f"⚠️  {w}")

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
        print(f"   {hist_path}  (+ .tif, .png)")
        print(f"   {stats_path}")

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
    p.add_argument("--seed-h", type=float, default=2.0, dest="seed_h",
                   help="Seed depth in pixels for watershed (h-maxima of the "
                        "distance transform); raise if particles are over-split, "
                        "lower if touching particles stay merged (default: 2.0)")
    p.add_argument("--fg-distance-ratio", type=float, default=None,
                   help=argparse.SUPPRESS)  # deprecated, ignored
    polarity = p.add_mutually_exclusive_group()
    polarity.add_argument("--bright-particles", action="store_true",
                          help="Force bright particles on dark background")
    polarity.add_argument("--dark-particles", action="store_true",
                          help="Force dark particles on bright background")
    p.add_argument("--keep-border", action="store_true",
                   help="Keep particles touching the image border")
    p.add_argument("--crop-top", type=int, default=0, dest="crop_top",
                   help="Pixels to crop from the top of the image "
                        "(default: 0)")
    p.add_argument("--crop-bottom", type=int, default=0, dest="crop_bottom",
                   help="Pixels to crop from the bottom of the image, "
                        "useful for stripping SEM info bars (default: 0)")
    p.add_argument("--save-debug", action="store_true", dest="save_debug",
                   help="Also save a 2x2 diagnostic panel showing the "
                        "intermediate stages of segmentation")
    p.add_argument("--journal", choices=sorted(JOURNALS), default="acs",
                   help="Journal format for the size histogram (default: acs)")
    p.add_argument("--fig-width", choices=["single", "onehalf", "double"],
                   default="single", dest="fig_width",
                   help="Column width of the histogram (default: single; "
                        "onehalf is Elsevier only)")
    p.add_argument("--no-lognormal", action="store_true", dest="no_lognormal",
                   help="Do not overlay the log-normal fit on the histogram")
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
        seed_h=args.seed_h,
        fg_distance_ratio=args.fg_distance_ratio,
        particles_are_bright=bright,
        keep_border=args.keep_border,
        crop_top_px=args.crop_top,
        crop_bottom_px=args.crop_bottom,
        save_debug=args.save_debug,
        journal=args.journal,
        fig_width=args.fig_width,
        lognormal_fit=not args.no_lognormal,
    )


if __name__ == "__main__":
    main()
