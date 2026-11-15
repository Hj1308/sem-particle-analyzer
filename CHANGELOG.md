# Changelog

## v0.2.0 (2026-11-15)

### Added
- `pyproject.toml`: the tool installs with `pip install -e .` and provides the
  `sem-particle-analyzer` command.
- Test suite on synthetic images with known geometry (`tests/`), and GitHub
  Actions CI on Python 3.9, 3.11 and 3.13.

- **Journal-ready size histogram.** `<name>_size_distribution` is now written
  as PDF (editable text), TIFF and PNG at the printed column width of ACS,
  RSC or Elsevier (`--journal`, `--fig-width`), with Freedman–Diaconis bins,
  an optional log-normal fit (`--no-lognormal`) and N, mean ± SD (GUM-rounded)
  and d₁₀/d₅₀/d₉₀ on the panel.
- **`<name>_size_statistics.csv`** and `size_statistics()`: N, mean, SD, 95 %
  CI of the mean (t), d₁₀/d₅₀/d₉₀, span, log-normal geometric mean and SD. A
  warning is printed below 100 particles.
- `scipy` is now a dependency (t quantile, log-normal curve).

### Changed
- **Segmentation (v0.2).** Seeds are the h-maxima of the Euclidean distance
  transform (`h`, CLI `--seed-h`, default 2 px) instead of one global threshold
  `dist > 0.4 · max(dist)`; every connected component without a seed gets one at
  its own maximum. The watershed runs on −distance inside the particle mask
  (scikit-image), so every foreground pixel belongs to a particle. Polarity is
  decided from the image frame (the background dominates the border).
  Perimeter uses the Crofton estimate. `fg_distance_ratio` /
  `--fg-distance-ratio` are deprecated and ignored (warning). New dependency:
  `scikit-image`.
- `measure_particles` uses `regionprops` (one pass instead of one full-image scan
  per particle).

### Fixed
All seven defects below were pinned as strict xfails in the previous change and
now pass as regression tests:
- Small particles were dropped next to large ones (6 large + 12 small disks →
  6 found; 400 log-normal disks → 193 found, d₅₀ +22 %). Now 18/18 and 400/400.
- Two disks overlapping by 5 px were not split at the default setting.
- Every particle lost its one-pixel watershed boundary ring (equivalent diameter
  −6.3 % at 16 px … −1.1 % at 120 px). Now within 0.3 %.
- A perfect disk had circularity 0.91–0.95. Now 1.01–1.02 for 30–120 px disks
  (pixelation still lowers it for particles below ~20 px).
- Auto polarity failed when particles covered more than half of the frame.
- Edge-clipped particles were never dropped (`cv2.watershed` marked the image
  frame as boundary).
- A featureless frame yielded a fake particle.

### Tests and CI
- The polydisperse-scene fixture is module-scoped (pytest 9 deprecates class-scoped
  fixtures defined as instance methods); CI also runs Python 3.14.

### Docs
- `CITATION.cff` added.
- README: validation table on synthetic scenes; the old tuning guide had the
  direction of `--fg-distance-ratio` reversed — replaced by `--seed-h`.
