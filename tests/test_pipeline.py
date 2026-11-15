"""Tests on synthetic images with known geometry.

Classes Test* check behaviour that is correct today. TestKnownDefects pins the four
defects found in the 2026-11-15 review as strict xfails: each must start passing
(and its xfail marker be removed) when the corresponding fix lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

import particle_analyzer as pa
from synthetic import disks, grid


def run(img, bright=True, pixel_size_nm=1.0, min_area_px=10, **seg):
    pre = pa.preprocess(pa.normalize_to_8bit(img))
    labels, _ = pa.segment(pre, particles_are_bright=bright, **seg)
    return pa.measure_particles(labels, pixel_size_nm=pixel_size_nm, min_area_px=min_area_px)


# -- I/O ----------------------------------------------------------------------------
class TestIO:
    def test_16bit_tiff_is_read_unchanged(self, tmp_path):
        img = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16).astype(np.uint16)
        p = tmp_path / "a.tif"
        tifffile.imwrite(p, img)
        out = pa.read_image(str(p))
        assert out.dtype == np.uint16 and np.array_equal(out, img)

    def test_multipage_tiff_returns_first_page(self, tmp_path):
        stack = np.stack([np.full((32, 40), v, np.uint8) for v in (10, 20, 30)])
        p = tmp_path / "stack.tif"
        tifffile.imwrite(p, stack, photometric="minisblack")
        out = pa.read_image(str(p))
        assert out.shape == (32, 40) and out[0, 0] == 10

    def test_rgb_png_becomes_grayscale(self, tmp_path):
        import cv2

        rgb = np.zeros((20, 30, 3), np.uint8)
        rgb[..., 1] = 255
        p = tmp_path / "c.png"
        cv2.imwrite(str(p), rgb)
        out = pa.read_image(str(p))
        assert out.ndim == 2 and out.shape == (20, 30)

    def test_missing_png_raises(self, tmp_path):
        with pytest.raises(IOError):
            pa.read_image(str(tmp_path / "nope.png"))

    def test_normalize_constant_image(self):
        out = pa.normalize_to_8bit(np.full((10, 10), 7, np.uint16))
        assert out.dtype == np.uint8 and np.all(out == 0)

    def test_normalize_stretches_16bit_range(self):
        img = np.linspace(1000, 5000, 10000).reshape(100, 100).astype(np.uint16)
        out = pa.normalize_to_8bit(img)
        assert out.min() == 0 and out.max() == 255


class TestPreprocess:
    def test_even_kernel_is_made_odd(self):
        img = np.random.default_rng(0).integers(0, 255, (50, 50), dtype=np.uint8)
        assert pa.preprocess(img, gaussian_k=4).shape == img.shape

    def test_clahe_and_blur_can_be_disabled(self):
        img = np.random.default_rng(0).integers(0, 255, (50, 50), dtype=np.uint8)
        assert np.array_equal(pa.preprocess(img, clahe_clip=0, gaussian_k=1), img)


# -- segmentation and measurement (correct today) ----------------------------------
class TestSegmentation:
    def test_counts_separated_equal_disks(self):
        spec = grid(r=20, pitch=80)
        img, _ = disks(spec)
        assert len(run(img)) == len(spec)

    def test_splits_two_touching_disks_with_tuned_seed_ratio(self):
        img, _ = disks([(250, 300, 40), (325, 300, 40)])  # centres 75 px apart, overlap 5 px
        assert len(run(img, fg_distance_ratio=0.7)) == 2

    def test_dark_particles_with_explicit_polarity(self):
        spec = grid(r=20, pitch=80)
        img, _ = disks(spec, bright=False)
        assert len(run(img, bright=False)) == len(spec)

    def test_noise_does_not_create_particles(self):
        spec = grid(r=20, pitch=80)
        img, _ = disks(spec, noise_sd=8)
        assert len(run(img, min_area_px=30)) == len(spec)


class TestMeasurement:
    def test_calibration_scales_lengths_and_areas(self):
        img, _ = disks([(300, 300, 30)])
        a, b = run(img, pixel_size_nm=1.0), run(img, pixel_size_nm=2.5)
        assert b.eq_diameter_nm.iloc[0] == pytest.approx(2.5 * a.eq_diameter_nm.iloc[0])
        assert b.area_nm2.iloc[0] == pytest.approx(6.25 * a.area_nm2.iloc[0])
        assert b.eq_diameter_px.iloc[0] == a.eq_diameter_px.iloc[0]

    def test_keep_border_keeps_clipped_particle(self):
        img, _ = disks([(300, 300, 30), (10, 300, 30)])  # second one is clipped by the edge
        pre = pa.preprocess(pa.normalize_to_8bit(img))
        labels, _ = pa.segment(pre, particles_are_bright=True)
        assert len(pa.measure_particles(labels, min_area_px=10, touch_border=True)) == 2

    def test_area_filters(self):
        img, _ = disks([(150, 150, 10), (400, 400, 40)])
        pre = pa.preprocess(pa.normalize_to_8bit(img))
        labels, _ = pa.segment(pre, particles_are_bright=True, fg_distance_ratio=0.2)
        assert len(pa.measure_particles(labels, min_area_px=1000)) == 1
        assert len(pa.measure_particles(labels, min_area_px=10, max_area_px=1000)) == 1

    def test_ellipse_aspect_ratio(self):
        import cv2

        img = np.full((400, 400), 40, np.uint8)
        cv2.ellipse(img, (200, 200), (80, 40), 30, 0, 360, 200, -1)
        df = run(img)
        assert len(df) == 1
        assert df.aspect_ratio.iloc[0] == pytest.approx(2.0, rel=0.05)
        assert df.major_axis_nm.iloc[0] == pytest.approx(160, rel=0.05)


# -- end-to-end and CLI ----------------------------------------------------------------
class TestAnalyze:
    def test_writes_outputs(self, tmp_path):
        img, _ = disks(grid(r=20, pitch=80))
        p = tmp_path / "sample.tif"
        tifffile.imwrite(p, img)
        df = pa.analyze(
            str(p),
            output_dir=str(tmp_path / "out"),
            pixel_size_nm=2.0,
            save_debug=True,
            verbose=False,
        )
        assert len(df) == len(grid(r=20, pitch=80))
        for suffix in (
            "_annotated.tif",
            "_measurements.csv",
            "_size_distribution.png",
            "_debug.png",
        ):
            assert (tmp_path / "out" / f"sample{suffix}").exists()
        csv = pd.read_csv(tmp_path / "out" / "sample_measurements.csv")
        assert len(csv) == len(df)

    def test_crop_removes_info_bar(self, tmp_path):
        img, _ = disks(grid(r=20, pitch=80, shape=(500, 600)), shape=(500, 600))
        bar = np.full((100, 600), 230, np.uint8)  # bright SEM info bar
        bar[40:60, 50:550] = 20
        full = np.vstack([img, bar])
        p = tmp_path / "bar.tif"
        tifffile.imwrite(p, full)
        df = pa.analyze(
            str(p),
            output_dir=str(tmp_path),
            crop_bottom_px=100,
            particles_are_bright=True,
            verbose=False,
        )
        assert len(df) == len(grid(r=20, pitch=80, shape=(500, 600)))

    def test_no_particles_after_filtering_returns_empty(self, tmp_path):
        img, _ = disks([(50, 50, 3)], shape=(100, 100))
        p = tmp_path / "tiny.tif"
        tifffile.imwrite(p, img)
        df = pa.analyze(
            str(p),
            output_dir=str(tmp_path),
            min_area_px=500,
            particles_are_bright=True,
            verbose=False,
        )
        assert isinstance(df, pd.DataFrame) and df.empty


class TestCLI:
    def test_defaults(self):
        a = pa._build_argparser().parse_args(["x.tif"])
        assert (a.pixel_size, a.min_area, a.fg_distance_ratio, a.keep_border) == (
            1.0,
            30,
            0.4,
            False,
        )

    def test_polarity_flags_are_exclusive(self):
        with pytest.raises(SystemExit):
            pa._build_argparser().parse_args(["x.tif", "--bright-particles", "--dark-particles"])

    def test_main_runs(self, tmp_path, monkeypatch):
        img, _ = disks(grid(r=20, pitch=80))
        p = tmp_path / "cli.tif"
        tifffile.imwrite(p, img)
        monkeypatch.setattr(
            "sys.argv",
            ["sem-particle-analyzer", str(p), "-o", str(tmp_path / "o"), "--bright-particles"],
        )
        pa.main()
        assert (tmp_path / "o" / "cli_measurements.csv").exists()


# -- known defects (review 2026-11-15) -------------------------------------------------
class TestKnownDefects:
    @pytest.mark.xfail(reason="global seed threshold 0.4*max(dist) drops small particles")
    def test_small_particles_kept_in_polydisperse_sample(self):
        spec = [(80 + i * 90, 100, 40) for i in range(6)] + [
            (60 + i * 45, 350, 10) for i in range(12)
        ]
        img, _ = disks(spec)
        assert len(run(img)) == 18

    @pytest.mark.xfail(reason="watershed boundary ring (-1) is removed from every particle")
    @pytest.mark.parametrize("r", [8, 15, 30, 60])
    def test_equivalent_diameter_within_one_percent(self, r):
        img, true_d = disks([(300, 300, r)])
        assert run(img).eq_diameter_px.iloc[0] == pytest.approx(true_d[0], rel=0.01)

    @pytest.mark.xfail(reason="cv2.arcLength on the pixel contour overestimates the perimeter")
    @pytest.mark.parametrize("r", [15, 30, 60])
    def test_disk_circularity_close_to_one(self, r):
        img, _ = disks([(300, 300, r)])
        assert run(img).circularity.iloc[0] == pytest.approx(1.0, abs=0.03)

    @pytest.mark.xfail(reason="auto polarity assumes particles cover < 50 % of the frame")
    def test_auto_polarity_with_dense_particles(self):
        spec = grid(r=28, pitch=58, margin=30)
        img, true_d = disks(spec)
        assert (img > 100).mean() > 0.5
        pre = pa.preprocess(pa.normalize_to_8bit(img))
        labels, _ = pa.segment(pre)  # polarity auto-detected
        df = pa.measure_particles(labels, min_area_px=10)
        assert df.eq_diameter_px.median() == pytest.approx(np.median(true_d), rel=0.1)

    @pytest.mark.xfail(reason="seed ratio 0.4 does not split two disks overlapping by 5 px")
    def test_splits_two_touching_disks_at_default_settings(self):
        img, _ = disks([(250, 300, 40), (325, 300, 40)])
        assert len(run(img)) == 2

    @pytest.mark.xfail(
        reason="cv2.watershed sets the image frame to -1, so no label reaches row/col 0 "
        "and edge-clipped particles are never dropped"
    )
    def test_border_particles_are_dropped_by_default(self):
        img, _ = disks([(300, 300, 30), (10, 300, 30)])
        pre = pa.preprocess(pa.normalize_to_8bit(img))
        labels, _ = pa.segment(pre, particles_are_bright=True)
        assert len(pa.measure_particles(labels, min_area_px=10)) == 1

    @pytest.mark.xfail(reason="Otsu on a featureless frame still splits it, giving a fake particle")
    def test_blank_image_has_no_particles(self):
        assert run(np.full((100, 100), 40, np.uint8)).empty
