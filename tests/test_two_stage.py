"""B11: unit tests for two_stage.py partition and table builder (no model)."""

from __future__ import annotations

from benchmarks.two_stage import (
    _build_markdown_table,
    _percentile,
    _summarize,
    group_description,
    partition_options,
)


class TestPartition:
    def test_255_options_into_16_groups(self):
        """255 options / 16 = 15 full groups + 1 partial (15 options)."""
        choices = [f"cat_{i:03d}" for i in range(255)]
        groups = partition_options(choices, group_size=16)
        # 255 / 16 = 15.9375 -> 16 groups (last has 15).
        assert len(groups) == 16
        # First 15 groups have 16, last has 15.
        for _, group in groups[:-1]:
            assert len(group) == 16
        assert len(groups[-1][1]) == 15
        # Every option appears exactly once.
        all_opts = [opt for _, group in groups for opt in group]
        assert len(all_opts) == 255
        assert set(all_opts) == set(choices)

    def test_partition_is_deterministic_and_sorted(self):
        """Same input always produces the same sorted partition."""
        choices = ["zebra", "apple", "mango", "banana", "cherry"]
        groups1 = partition_options(choices, group_size=2)
        groups2 = partition_options(choices, group_size=2)
        assert groups1 == groups2
        # Options are sorted within the partition.
        all_opts = [opt for _, group in groups1 for opt in group]
        assert all_opts == sorted(choices)

    def test_group_size_1(self):
        """group_size=1 produces one group per option."""
        choices = ["a", "b", "c"]
        groups = partition_options(choices, group_size=1)
        assert len(groups) == 3
        assert all(len(g) == 1 for _, g in groups)

    def test_partial_last_group(self):
        """11 options / group_size 4 = 2 full + 1 partial (3)."""
        choices = list("abcdefghijk")
        groups = partition_options(choices, group_size=4)
        assert len(groups) == 3
        assert len(groups[0][1]) == 4
        assert len(groups[1][1]) == 4
        assert len(groups[2][1]) == 3

    def test_group_indices_are_sequential(self):
        choices = list("abcdefgh")
        groups = partition_options(choices, group_size=3)
        indices = [gidx for gidx, _ in groups]
        assert indices == [0, 1, 2]


class TestGroupDescription:
    def test_single_option_group(self):
        group = ["solo"]
        desc = group_description(group)
        assert "solo" in desc

    def test_range_description(self):
        group = ["alpha", "beta", "gamma"]
        desc = group_description(group)
        assert "alpha" in desc
        assert "gamma" in desc

    def test_multi_option_group_has_first_and_last(self):
        group = ["a", "b", "c", "d", "e"]
        desc = group_description(group)
        assert "a" in desc
        assert "e" in desc
        assert "b" not in desc  # only first and last


class TestSummarize:
    def test_accuracy_calculation(self):
        results = [
            {"correct": True, "wall_ms": 100.0},
            {"correct": False, "wall_ms": 200.0},
            {"correct": True, "wall_ms": 150.0},
            {"correct": True, "wall_ms": 300.0},
        ]
        summary = _summarize(results)
        assert summary["n"] == 4
        assert summary["accuracy"] == 0.75  # 3/4
        assert summary["wall_ms_median"] == 175.0  # median of [100, 200, 150, 300]

    def test_stage1_error_rate(self):
        results = [
            {"correct": True, "stage1_correct": True, "wall_ms": 100.0},
            {"correct": False, "stage1_correct": False, "wall_ms": 200.0},
            {"correct": True, "stage1_correct": True, "wall_ms": 150.0},
        ]
        summary = _summarize(results)
        # stage1 error rate = 1 - (2/3) = 0.3333
        assert abs(summary["stage1_error_rate"] - 0.3333) < 0.01

    def test_empty_results(self):
        summary = _summarize([])
        assert summary["n"] == 0
        assert summary["accuracy"] == 0.0
        assert summary["wall_ms_median"] == 0


class TestPercentile:
    def test_p50_is_median(self):
        data = [10, 20, 30, 40, 50]
        assert _percentile(data, 50) == 30

    def test_p95_of_small_list(self):
        data = [100, 200, 300]
        # p95 interpolates between 200 and 300
        p95 = _percentile(data, 95)
        assert 290 <= p95 <= 300

    def test_empty(self):
        assert _percentile([], 95) == 0.0


class TestMarkdownTable:
    def test_table_has_both_variants(self):
        one = {"accuracy": 0.5, "wall_ms_median": 100, "wall_ms_p95": 200}
        two = {"accuracy": 0.6, "wall_ms_median": 80, "wall_ms_p95": 150, "stage1_error_rate": 0.1}
        table = _build_markdown_table(one, two, "fast", "arm64-32gb", 12)
        assert "One-stage" in table
        assert "Two-stage" in table
        assert "50.0%" in table
        assert "60.0%" in table
        assert "arm64-32gb" in table
        assert "fast" in table
        assert "smoke on 0.5B" in table

    def test_table_includes_stage1_error_rate_column(self):
        one = {"accuracy": 0.5, "wall_ms_median": 100, "wall_ms_p95": 200}
        two = {"accuracy": 0.6, "wall_ms_median": 80, "wall_ms_p95": 150, "stage1_error_rate": 0.25}
        table = _build_markdown_table(one, two, "test", "arm64-16gb", 6)
        assert "Stage-1 error rate" in table
        assert "25.0%" in table
