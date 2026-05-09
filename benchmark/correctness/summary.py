"""Formatted correctness report output."""
from __future__ import annotations

from benchmark.common.table import format_float, print_section, print_table


def build_summary(rows: list[dict]) -> dict:
    passed = sum(1 for row in rows if row.get("verdict") == "PASS")
    failed = sum(1 for row in rows if row.get("verdict") == "FAIL")
    blocked = sum(1 for row in rows if row.get("verdict") == "BLOCKED")
    return {
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "ok": failed == 0,
    }


def _value(row: dict, key: str, default: str = "-") -> str:
    value = row.get(key, default)
    if value is None:
        return "-"
    return str(value)


def _metric(row: dict, key: str) -> str:
    return format_float(row.get(key))


def _print_environment(env: dict) -> None:
    print_section("Environment")
    print_table(
        ["Field", "Value"],
        [
            ["GPU", env["gpu"]],
            ["Compute Capability", env["compute_capability"]],
            ["CUDA/HIP", env["cuda_or_hip"]],
            ["Torch", env["torch"]],
            ["Triton", env["triton"]],
            ["Python", env["python"]],
            ["Platform", env["platform"]],
            ["BF16 Supported", str(env["bf16_supported"])],
        ],
    )


def _print_forward(rows: list[dict]) -> None:
    print_section("Forward Correctness")
    print_table(
        [
            "Path",
            "Shape",
            "Dtype",
            "Casting",
            "Offset",
            "Weight",
            "Mode",
            "Reference",
            "Max Abs",
            "Max Rel",
            "Tolerance",
            "Verdict",
            "Notes",
        ],
        [
            [
                _value(row, "path"),
                _value(row, "shape"),
                _value(row, "dtype"),
                _value(row, "casting"),
                _value(row, "offset"),
                _value(row, "weight"),
                _value(row, "mode"),
                _value(row, "reference"),
                _metric(row, "max_abs"),
                _metric(row, "max_rel"),
                _value(row, "tolerance"),
                _value(row, "verdict"),
                _value(row, "notes", ""),
            ]
            for row in rows
        ],
    )


def _print_backward(rows: list[dict]) -> None:
    print_section("Backward Correctness")
    print_table(
        [
            "Path",
            "Shape",
            "Dtype",
            "Casting",
            "Offset",
            "Weight",
            "In-place",
            "Cache rstd",
            "Mode",
            "Reference",
            "dx Abs",
            "dx Rel",
            "dw Abs",
            "dw Rel",
            "Tolerance",
            "Verdict",
            "Notes",
        ],
        [
            [
                _value(row, "path"),
                _value(row, "shape"),
                _value(row, "dtype"),
                _value(row, "casting"),
                _value(row, "offset"),
                _value(row, "weight"),
                _value(row, "in_place"),
                _value(row, "cache_rstd"),
                _value(row, "mode"),
                _value(row, "reference"),
                _metric(row, "dx_abs"),
                _metric(row, "dx_rel"),
                _metric(row, "dw_abs"),
                _metric(row, "dw_rel"),
                _value(row, "tolerance"),
                _value(row, "verdict"),
                _value(row, "notes", ""),
            ]
            for row in rows
        ],
    )


def _print_internal(rows: list[dict]) -> None:
    print_section("Path And Option Checks")
    print_table(
        [
            "Check",
            "Path",
            "Shape",
            "Dtype",
            "Casting",
            "Offset",
            "Reference",
            "Compared",
            "Max Abs",
            "Max Rel",
            "dx Abs",
            "dw Abs",
            "Tolerance",
            "Verdict",
            "Notes",
        ],
        [
            [
                _value(row, "check"),
                _value(row, "path"),
                _value(row, "shape"),
                _value(row, "dtype"),
                _value(row, "casting"),
                _value(row, "offset"),
                _value(row, "reference"),
                _value(row, "compared"),
                _metric(row, "max_abs"),
                _metric(row, "max_rel"),
                _metric(row, "dx_abs"),
                _metric(row, "dw_abs"),
                _value(row, "tolerance"),
                _value(row, "verdict"),
                _value(row, "notes", ""),
            ]
            for row in rows
        ],
    )


def _print_fold(rows: list[dict]) -> None:
    print_section("Gamma Fold Checks")
    print_table(
        [
            "Check",
            "Shape",
            "Dtype",
            "Reference",
            "Compared",
            "Max Abs",
            "Max Rel",
            "Tolerance",
            "Verdict",
            "Notes",
        ],
        [
            [
                _value(row, "check"),
                _value(row, "shape"),
                _value(row, "dtype"),
                _value(row, "reference"),
                _value(row, "compared"),
                _metric(row, "max_abs"),
                _metric(row, "max_rel"),
                _value(row, "tolerance"),
                _value(row, "verdict"),
                _value(row, "notes", ""),
            ]
            for row in rows
        ],
    )


def _print_blocked(rows: list[dict]) -> None:
    print_section("Blocked Checks")
    print_table(
        ["Check", "Reference", "Compared", "Verdict", "Notes"],
        [
            [
                _value(row, "check"),
                _value(row, "reference"),
                _value(row, "compared"),
                _value(row, "verdict"),
                _value(row, "notes", ""),
            ]
            for row in rows
        ],
    )


def _print_failures(rows: list[dict]) -> None:
    failures = [row for row in rows if row.get("verdict") == "FAIL"]
    if not failures:
        return
    print_section("Failures")
    print_table(
        ["Group", "Check", "Path", "Shape", "Dtype", "Casting", "Offset", "Reference", "Compared", "Max Abs", "Max Rel", "Notes"],
        [
            [
                _value(row, "group"),
                _value(row, "check"),
                _value(row, "path"),
                _value(row, "shape"),
                _value(row, "dtype"),
                _value(row, "casting"),
                _value(row, "offset"),
                _value(row, "reference"),
                _value(row, "compared"),
                _metric(row, "max_abs"),
                _metric(row, "max_rel"),
                _value(row, "notes", ""),
            ]
            for row in failures
        ],
    )


def print_correctness_report(env: dict, rows: list[dict], summary: dict) -> None:
    _print_environment(env)

    print_section("Summary")
    print_table(
        ["Total", "PASS", "FAIL", "BLOCKED", "Status"],
        [
            [
                summary["total"],
                summary["passed"],
                summary["failed"],
                summary["blocked"],
                "PASS" if summary["ok"] else "FAIL",
            ]
        ],
    )

    groups = {
        "forward": [row for row in rows if row.get("group") == "forward"],
        "backward": [row for row in rows if row.get("group") == "backward"],
        "internal": [row for row in rows if row.get("group") == "internal"],
        "fold": [row for row in rows if row.get("group") == "fold"],
        "blocked": [row for row in rows if row.get("group") == "blocked"],
    }
    _print_forward(groups["forward"])
    _print_backward(groups["backward"])
    _print_internal(groups["internal"])
    _print_fold(groups["fold"])
    _print_blocked(groups["blocked"])
    _print_failures(rows)
