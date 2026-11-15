"""Synthetic SEM-like images with known particle geometry (ground truth for tests)."""

from __future__ import annotations

import cv2
import numpy as np

BG, FG = 40, 200


def disks(spec, shape=(600, 600), bright=True, noise_sd=0.0, seed=0):
    """Filled disks on a uniform background.

    spec: iterable of (cx, cy, r) in pixels. Returns (image uint8, true eq. diameters in px),
    the diameters computed from the rasterised pixel areas, so they are the exact targets.
    """
    fg, bg = (FG, BG) if bright else (BG, FG)
    img = np.full(shape, bg, np.uint8)
    diam = []
    for cx, cy, r in spec:
        m = np.zeros(shape, np.uint8)
        cv2.circle(m, (int(cx), int(cy)), int(r), 1, -1)
        img[m == 1] = fg
        diam.append(2.0 * np.sqrt(m.sum() / np.pi))
    if noise_sd > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise_sd, shape), 0, 255).astype(np.uint8)
    return img, np.array(diam)


def grid(r, pitch, shape=(600, 600), margin=None):
    """Disks of radius r on a square grid; margin defaults to pitch/2 (all inside)."""
    m = pitch // 2 if margin is None else margin
    return [(x, y, r) for x in range(m, shape[1] - r, pitch) for y in range(m, shape[0] - r, pitch)]


def random_scene(n=400, shape=(1024, 1024), gm=12.0, gsd=1.35, gap=2.0, seed=1):
    """Non-overlapping disks with log-normal radii (3-40 px), at least `gap` px apart."""
    rng = np.random.default_rng(seed)
    spec, tries = [], 0
    while len(spec) < n and tries < 20000:
        tries += 1
        r = float(np.clip(rng.lognormal(np.log(gm), np.log(gsd)), 3, 40))
        x = rng.uniform(r + 2, shape[1] - r - 2)
        y = rng.uniform(r + 2, shape[0] - r - 2)
        if all(np.hypot(x - a, y - b) > r + c + gap for a, b, c in spec):
            spec.append((x, y, r))
    return [(int(round(x)), int(round(y)), int(round(r))) for x, y, r in spec]


def sem_like(img, blur_sd=1.0, gradient=0.4, noise_sd=10.0, seed=3):
    """Gaussian blur, left-to-right illumination gradient and Gaussian noise."""
    out = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur_sd)
    out = out * (1 - gradient / 2 + gradient * np.linspace(0, 1, img.shape[1])[None, :])
    out = out + np.random.default_rng(seed).normal(0, noise_sd, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
