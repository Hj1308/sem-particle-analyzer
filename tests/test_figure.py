"""Publication histogram and size statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

import particle_analyzer as pa
from synthetic import disks, grid


def _lognormal(n, gm=24.0, gsd=np.exp(0.25), seed=4):
    return np.random.default_rng(seed).lognormal(np.log(gm), np.log(gsd), n)


class TestStatistics:
    def test_percentiles_mean_and_ci(self):
        x = _lognormal(500)
        st = pa.size_statistics(x)
        assert st["N"] == 500
        assert st["d10"] == pytest.approx(np.percentile(x, 10))
        assert st["d50"] == pytest.approx(np.median(x))
        assert st["d90"] == pytest.approx(np.percentile(x, 90))
        lo, hi = st["ci95_mean"]
        assert lo < st["mean"] < hi
        assert hi - lo == pytest.approx(2 * 1.9647 * x.std(ddof=1) / np.sqrt(500), rel=1e-3)

    def test_lognormal_parameters_recovered(self):
        st = pa.size_statistics(_lognormal(5000))
        assert st["geo_mean"] == pytest.approx(24.0, rel=0.02)
        assert st["geo_sd"] == pytest.approx(np.exp(0.25), rel=0.02)

    def test_non_positive_and_nan_ignored(self):
        st = pa.size_statistics([np.nan, -1, 0, 10, 20])
        assert st["N"] == 2 and st["mean"] == 15

    def test_empty(self):
        assert pa.size_statistics([]) == {"N": 0}

    @pytest.mark.parametrize(
        "v, u, expected",
        [
            (12.34567, 0.02345, ("12.346", "0.023")),
            (1234, 250, ("1230", "250")),
            (52.346, 3.21, ("52", "3")),
            (1.2345, 0.096, ("1.23", "0.10")),
        ],
    )
    def test_gum_rounding(self, v, u, expected):
        assert pa._round_pm(v, u) == expected


class TestPublicationHistogram:
    @pytest.mark.parametrize(
        "journal, width, cm",
        [("acs", "single", 8.46), ("rsc", "double", 17.1), ("elsevier", "onehalf", 14.0)],
    )
    def test_printed_width_and_files(self, tmp_path, journal, width, cm):
        df = pd.DataFrame({"eq_diameter_nm": _lognormal(300)})
        st = pa.plot_size_distribution_publication(
            df, str(tmp_path / "h"), journal=journal, width=width
        )
        for ext in ("pdf", "tif", "png"):
            assert (tmp_path / f"h.{ext}").exists()
        dpi = pa.JOURNALS[journal]["dpi"]
        im = Image.open(tmp_path / "h.tif")
        assert im.size[0] / dpi * 2.54 == pytest.approx(cm, abs=0.02)
        assert im.info["dpi"][0] == pytest.approx(dpi)
        assert st["warnings"] == []

    def test_pdf_keeps_text_editable(self, tmp_path):
        df = pd.DataFrame({"eq_diameter_nm": _lognormal(300)})
        pa.plot_size_distribution_publication(df, str(tmp_path / "h"))
        raw = (tmp_path / "h.pdf").read_bytes()
        assert b"/FontFile2" in raw  # TrueType embedded (fonttype 42), not outlines

    def test_small_n_warns(self, tmp_path):
        df = pd.DataFrame({"eq_diameter_nm": _lognormal(40)})
        st = pa.plot_size_distribution_publication(df, str(tmp_path / "h"))
        assert any("Only 40 particles" in w for w in st["warnings"])

    def test_bad_journal_or_width(self, tmp_path):
        df = pd.DataFrame({"eq_diameter_nm": _lognormal(50)})
        with pytest.raises(ValueError):
            pa.plot_size_distribution_publication(df, str(tmp_path / "h"), journal="nature")
        with pytest.raises(ValueError):
            pa.plot_size_distribution_publication(
                df, str(tmp_path / "h"), journal="acs", width="onehalf"
            )


class TestPipelineOutputs:
    def test_analyze_writes_statistics_and_publication_figure(self, tmp_path):
        img, true_d = disks(grid(r=20, pitch=80))
        p = tmp_path / "s.tif"
        tifffile.imwrite(p, img)
        pa.analyze(
            str(p),
            output_dir=str(tmp_path / "o"),
            journal="rsc",
            verbose=False,
            particles_are_bright=True,
        )
        stats = pd.read_csv(tmp_path / "o" / "s_size_statistics.csv")
        assert stats["N"].iloc[0] == len(true_d)
        assert stats["journal"].iloc[0] == "rsc"
        assert {"d10", "d50", "d90", "ci95_mean_low", "geo_sd"} <= set(stats.columns)
        for ext in ("pdf", "tif", "png"):
            assert (tmp_path / "o" / f"s_size_distribution.{ext}").exists()

    def test_cli_journal_options(self, tmp_path, monkeypatch):
        img, _ = disks(grid(r=20, pitch=80))
        p = tmp_path / "c.tif"
        tifffile.imwrite(p, img)
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                str(p),
                "-o",
                str(tmp_path / "o"),
                "--journal",
                "elsevier",
                "--fig-width",
                "double",
                "--no-lognormal",
                "--bright-particles",
            ],
        )
        pa.main()
        im = Image.open(tmp_path / "o" / "c_size_distribution.tif")
        assert im.size[0] / 1000 * 2.54 == pytest.approx(19.0, abs=0.02)
