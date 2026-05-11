"""Terminal and markdown reporting for isolation timing."""
from __future__ import annotations

from benchmark.common.table import format_float, print_section, print_table


TIMED_STATUSES = {"OK", "EXTRA_WORK_REFERENCE"}


def _shape_text(shape: dict) -> str:
    if shape.get("batch") is not None and shape.get("seq") is not None:
        return f"b{shape['batch']} s{shape['seq']} h{shape['n']}"
    return f"m{shape['m']} n{shape['n']}"


def _latency(row: dict, key: str = "median") -> str:
    if not row.get("stats_ms"):
        return "-"
    return format_float((row.get("stats_ms") or {}).get(key))


def _speedup(row: dict, key: str) -> str:
    value = row.get(key)
    return "-" if value is None else f"{value:.3f}x"


def _bytes(value) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value >= 1024**3:
        return f"{value / 1024**3:.3f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.3f} MiB"
    return f"{int(value)} B"


def _availability_rows(availability: dict) -> list[list[object]]:
    return [
        [name, "OK" if item["available"] else "NOT_RUN", item["reason"]]
        for name, item in availability.items()
    ]


def _capability_rows(report: dict) -> list[list[object]]:
    return [
        [
            item["framework"],
            item["horizon_name"],
            item["status"],
            ",".join(item.get("expected_grads") or ()) or "-",
            ",".join(item.get("observed_grads") or ()) or "-",
            item.get("reason") or "",
        ]
        for item in report.get("capability_matrix", [])
    ]


def _exclusion_rows(report: dict) -> list[list[object]]:
    return [
        [
            item["framework"],
            item["horizon_name"],
            item["status"],
            item.get("reason") or "",
            ",".join(item.get("expected_grads") or ()) or "-",
            ",".join(item.get("observed_grads") or ()) or "-",
        ]
        for item in report.get("fairness_exclusions", [])
    ]


def _cold_start_rows(report: dict) -> list[list[object]]:
    rows = []
    horizon_names = {item["id"]: item["name"] for item in report.get("horizons", [])}
    for row in report.get("cold_start", []):
        rows.append(
            [
                horizon_names.get(row["horizon_id"], row["horizon_id"]),
                row["framework"],
                row.get("shape_text") or _shape_text(row["shape"]),
                row["status"],
                format_float(row.get("elapsed_ms")),
                _bytes(row.get("vram_bytes")) if row.get("status") == "OK" else "-",
                row.get("notes") or "",
            ]
        )
    return rows


def _horizon_rows(report: dict, horizon_id: str) -> list[list[object]]:
    selected = [row for row in report["results"] if row["horizon_id"] == horizon_id]
    return [
        [
            row.get("shape_text") or _shape_text(row["shape"]),
            row["framework"],
            row["status"],
            _latency(row, "median"),
            _latency(row, "p95"),
            _speedup(row, "speedup_vs_pytorch"),
            _speedup(row, "speedup_vs_liger"),
            format_float(row.get("gbps")) if row.get("status") in TIMED_STATUSES else "-",
            format_float(row.get("peak_utilization_pct")) if row.get("status") in TIMED_STATUSES else "-",
            ",".join(row.get("observed_grads") or ()) or "-",
            row.get("path") or "-",
            row.get("exclusion_reason") or row.get("notes") or "",
        ]
        for row in selected
    ]


def _vram_rows(report: dict, horizon_id: str) -> list[list[object]]:
    return [
        [
            row.get("shape_text") or _shape_text(row["shape"]),
            row["framework"],
            row["status"],
            _bytes(row.get("vram_bytes")) if row.get("status") in TIMED_STATUSES else "-",
            row.get("exclusion_reason") or "",
        ]
        for row in report["results"]
        if row["horizon_id"] == horizon_id
    ]


def _profiling_rows(report: dict) -> list[list[object]]:
    return [
        [
            row["horizon_name"],
            row.get("shape_text") or _shape_text(row["shape"]),
            row["framework"],
            format_float(row.get("arithmetic_intensity")),
            format_float(row.get("gbps")),
            format_float(row.get("peak_utilization_pct")),
            row.get("roofline", "unknown"),
        ]
        for row in report["results"]
        if row.get("status") in TIMED_STATUSES
    ]


def _path_rows(report: dict) -> list[list[object]]:
    seen = set()
    rows = []
    for row in report["results"]:
        key = (row["horizon_name"], row["framework"], row.get("shape_group"), row.get("path"))
        if key in seen:
            continue
        seen.add(key)
        rows.append([row["horizon_name"], row["framework"], row.get("shape_group"), row.get("path")])
    return rows


def _horizon_meta(horizon: dict) -> str:
    eligible = ", ".join(horizon.get("eligible") or ()) or "-"
    extra = ", ".join(horizon.get("include_extra_work") or ()) or "-"
    return f"Measures: {horizon['description']} | Eligible: {eligible} | Extra-work references: {extra}"


def print_report(report: dict) -> None:
    env = report["environment"]
    settings = report["settings"]

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
            ["BF16 Supported", env["bf16_supported"]],
            ["Peak Profile", settings["peak_profile"]],
            ["Peak Bandwidth GB/s", format_float(settings["peak_bandwidth_gbps"])],
        ],
    )

    print_section("Run Settings")
    print_table(
        ["Field", "Value"],
        [
            ["Mode", report["mode"]],
            ["Dtype", report["dtype"]],
            ["Warmup", report["warmup"]],
            ["Runs", report["runs"]],
            ["Seed", settings["seed"]],
            ["Steady State", "CUDA events after warmup/autotune"],
            ["Cold Start", "fresh subprocess with temporary Triton cache"],
        ],
    )

    print_section("Competitor Availability")
    print_table(["Framework", "Status", "Notes"], _availability_rows(report["competitor_availability"]))

    print_section("Capability Matrix")
    print_table(["Framework", "Horizon", "Status", "Expected", "Observed", "Reason"], _capability_rows(report))

    print_section("Fairness Exclusions")
    print_table(["Framework", "Horizon", "Status", "Reason", "Expected", "Observed"], _exclusion_rows(report))

    print_section("Folding Canary")
    folding = report.get("folding_canary", {})
    print_table(["Status", "Max Abs"], [[folding.get("status", "-"), format_float(folding.get("max_abs"))]])

    print_section("Cold Start / Autotune")
    print_table(["Horizon", "Framework", "Shape", "Status", "Elapsed ms", "Peak", "Notes"], _cold_start_rows(report))

    print_section("Steady-State Timing")
    headers = [
        "Shape",
        "Framework",
        "Status",
        "Median ms",
        "P95 ms",
        "vs PyTorch",
        "vs Liger",
        "GB/s",
        "Peak %",
        "Observed",
        "Path",
        "Notes",
    ]
    for horizon in report["horizons"]:
        print_section(horizon["name"])
        print(_horizon_meta(horizon))
        print_table(headers, _horizon_rows(report, horizon["id"]))

    print_section("VRAM")
    for horizon in report["horizons"]:
        print_section(f"{horizon['name']} VRAM")
        print_table(["Shape", "Framework", "Status", "Peak", "Notes"], _vram_rows(report, horizon["id"]))

    print_section("Profiling")
    print_table(["Horizon", "Shape", "Framework", "AI", "GB/s", "Peak %", "Roofline"], _profiling_rows(report))

    print_section("Path Coverage")
    print_table(["Horizon", "Framework", "Shape Group", "Path"], _path_rows(report))

    print_section("Warnings")
    print_table(["Warning"], [[warning] for warning in report["warnings"]])


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_None_\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def markdown_report(report: dict) -> str:
    env = report["environment"]
    settings = report["settings"]
    parts = ["# RMSNorm Isolation Timing", ""]
    parts.extend(
        [
            "## Environment",
            _md_table(
                ["Field", "Value"],
                [
                    ["GPU", env["gpu"]],
                    ["Compute Capability", env["compute_capability"]],
                    ["CUDA/HIP", env["cuda_or_hip"]],
                    ["Torch", env["torch"]],
                    ["Triton", env["triton"]],
                    ["Python", env["python"]],
                    ["Platform", env["platform"]],
                    ["BF16 Supported", env["bf16_supported"]],
                    ["Peak Profile", settings["peak_profile"]],
                    ["Peak Bandwidth GB/s", format_float(settings["peak_bandwidth_gbps"])],
                ],
            ),
            "## Run Settings",
            _md_table(
                ["Field", "Value"],
                [
                    ["Mode", report["mode"]],
                    ["Dtype", report["dtype"]],
                    ["Warmup", report["warmup"]],
                    ["Runs", report["runs"]],
                    ["Seed", settings["seed"]],
                    ["Steady State", "CUDA events after warmup/autotune"],
                    ["Cold Start", "fresh subprocess with temporary Triton cache"],
                ],
            ),
            "## Competitor Availability",
            _md_table(["Framework", "Status", "Notes"], _availability_rows(report["competitor_availability"])),
            "## Capability Matrix",
            _md_table(["Framework", "Horizon", "Status", "Expected", "Observed", "Reason"], _capability_rows(report)),
            "## Fairness Exclusions",
            _md_table(["Framework", "Horizon", "Status", "Reason", "Expected", "Observed"], _exclusion_rows(report)),
            "## Folding Canary",
            _md_table(
                ["Status", "Max Abs"],
                [[report.get("folding_canary", {}).get("status", "-"), format_float(report.get("folding_canary", {}).get("max_abs"))]],
            ),
            "## Cold Start / Autotune",
            _md_table(["Horizon", "Framework", "Shape", "Status", "Elapsed ms", "Peak", "Notes"], _cold_start_rows(report)),
            "## Steady-State Timing",
        ]
    )
    headers = [
        "Shape",
        "Framework",
        "Status",
        "Median ms",
        "P95 ms",
        "vs PyTorch",
        "vs Liger",
        "GB/s",
        "Peak %",
        "Observed",
        "Path",
        "Notes",
    ]
    for horizon in report["horizons"]:
        parts.extend([f"### {horizon['name']}", _horizon_meta(horizon), _md_table(headers, _horizon_rows(report, horizon["id"]))])

    parts.append("## VRAM")
    for horizon in report["horizons"]:
        parts.extend([f"### {horizon['name']} VRAM", _md_table(["Shape", "Framework", "Status", "Peak", "Notes"], _vram_rows(report, horizon["id"]))])

    parts.extend(
        [
            "## Profiling",
            _md_table(["Horizon", "Shape", "Framework", "AI", "GB/s", "Peak %", "Roofline"], _profiling_rows(report)),
            "## Path Coverage",
            _md_table(["Horizon", "Framework", "Shape Group", "Path"], _path_rows(report)),
            "## Warnings",
            _md_table(["Warning"], [[warning] for warning in report["warnings"]]),
        ]
    )
    return "\n".join(parts)

