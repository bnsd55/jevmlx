"""W6-2 prep tests: benchmarks/probe.py — fake-model tests for the
table/JSON builders (real probes run in the M5 runbook behind --probe)."""

from benchmarks.probe import adapters_table_lines, probe_slope, slope_table_lines


class _FakeModel:
    """Minimal model for probe_slope: vocab/hidden only."""

    class _Args:
        vocab_size = 64
        hidden_size = 16

    args = _Args()


class TestProbeSlope:
    def test_produces_table_and_json_shape(self):
        """The slope probe returns the JSON-able shape and the table
        renders one line per width plus a header."""
        model = _FakeModel()
        result = probe_slope(model, widths=(4, 8), batches=(1, 2), hidden_size=16)
        assert set(result) == {
            "widths",
            "batches",
            "peak_bytes",
            "fitted_slope_per_bin",
            "vocab",
            "hidden",
        }
        assert result["widths"] == [4, 8]
        assert result["vocab"] == 64
        assert result["hidden"] == 16
        # peak_bytes keyed by str(width) -> str(batch) -> int
        assert set(result["peak_bytes"]) == {"4", "8"}
        assert set(result["peak_bytes"]["4"]) == {"1", "2"}
        for b in ("1", "2"):
            assert isinstance(result["peak_bytes"]["4"][b], int)
            assert result["peak_bytes"]["4"][b] >= 0

    def test_fitted_slope_is_linear_for_zeros(self):
        """mx.zeros slab + sum: the peak scales ~exactly with batch, so the
        fitted slope sits near 1.0 (within a small band)."""
        model = _FakeModel()
        result = probe_slope(model, widths=(4,), batches=(1, 2))
        assert 0.0 < result["fitted_slope_per_bin"]["4"] <= 4.0

    def test_table_lines_shape(self):
        result = probe_slope(_FakeModel(), widths=(4,), batches=(1, 2))
        lines = slope_table_lines(result)
        assert len(lines) == 2  # header + one width row
        assert "width" in lines[0]
        assert "slope" in lines[0]
        assert "4" in lines[1]


class TestAdaptersTable:
    def test_table_lines(self):
        result = {
            "cases": [
                {
                    "case": "alpha",
                    "max_abs_diff": 1.2e-7,
                    "full_width_ms": 0.4,
                    "decision_pos_ms": 0.1,
                }
            ],
            "max_abs_diff": 1.2e-7,
        }
        lines = adapters_table_lines(result)
        assert len(lines) == 3  # header + row + overall
        assert "overall max abs diff" in lines[-1]
        assert "alpha" in lines[1]
