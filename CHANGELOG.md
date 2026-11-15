# Changelog

## Unreleased

### Added
- `pyproject.toml`: the tool installs with `pip install -e .` and provides the
  `sem-particle-analyzer` command.
- Test suite on synthetic images with known geometry (`tests/`), and GitHub
  Actions CI on Python 3.9, 3.11 and 3.13.

### Known issues (pinned as strict xfail tests, to be fixed next)
Measured on synthetic disks with known size:
- A single seed threshold (0.4 × the largest distance-transform peak) drops small
  particles in polydisperse images: 6 large + 12 small disks → 6 found.
- At the default seed ratio two disks overlapping by 5 px are not split.
- Every particle loses its one-pixel watershed boundary ring: equivalent
  diameter −6.3 % at 16 px, −1.1 % at 120 px.
- Circularity of a perfect disk is 0.91–0.95 (perimeter from `cv2.arcLength`).
- Auto polarity fails when particles cover more than half of the frame (the
  background was measured as one particle).
- Edge-clipped particles are never dropped: `cv2.watershed` sets the image frame
  to −1, so no label reaches the first or last row/column.
- A featureless frame yields a fake particle.
