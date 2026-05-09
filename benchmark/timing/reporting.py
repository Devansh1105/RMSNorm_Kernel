"""Terminal and markdown reporting for isolation timing."""
from __future__ import annotations

from benchmark.common.table import format_float, print_section, print_table


SWEEP_TITLES = {
    "reference": "Reference Config",
    "seq": "Sequence Sweep",
    "batch": "Batch Sweep",
    "hidden": "Hidden Sweep",
    "qk_norm": "QK-Norm Sweep",
}


def _value(value, default: str = "-") -> str:
    if value is None:
        return default
    return str(value)


def _shape_text(shape: dict) -> str:
    batch = shape.get("batch")
    seq = shape.get("seq")
    m = shape.get("m")
    n = shape.get("n")
    if batch is not None and seq is not None:
        return f"b{batch} s{seq} h{n}"
    return f"m{m} n{n}"


def _latency(row: dict, key: str = "median") -> str:
    if row.get("status") != "OK":
        return "-"
    return format_float((row.get("stats_ms") or {}).get(key))


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


def _timing_rows(rows: list[dict], sweep: str) -> list[list[object]]:
    selected = [row for row in rows if row.get("sweep") == sweep]
    return [
        [
            row["framework"],
            row["pass"],
            _shape_text(row["shape"]),
            row["status"],
            _latency(row, "median"),
            _latency(row, "p95"),
            format_float(row.get("gbps")) if row.get("status") == "OK" else "-",
            format_float(row.get("peak_utilization_pct")) if row.get("status") == "OK" else "-",
            row.get("notes") or "",
        ]
        for row in selected
    ]


def _vram_rows(rows: list[dict]) -> list[list[object]]:
    return [
        [
            row["sweep"],
            row["framework"],
            row["pass"],
            _shape_text(row["shape"]),
            row["status"],
            _bytes(row.get("vram_bytes")) if row.get("status") == "OK" else "-",
        ]
        for row in rows
    ]


def _profiling_rows(rows: list[dict]) -> list[list[object]]:
    return [
        [
            row["sweep"],
            row["framework"],
            row["pass"],
            _shape_text(row["shape"]),
            format_float(row.get("arithmetic_intensity")),
            format_float(row.get("gbps")),
            format_float(row.get("peak_utilization_pct")),
            row.get("roofline", "unknown"),
        ]
        for row in rows
        if row.get("status") == "OK"
    ]


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
            ["Casting Mode", settings["casting_mode"]],
            ["Offset", settings["offset"]],
            ["Eps", settings["eps"]],
            ["Backward in_place", settings["backward_in_place"]],
        ],
    )

    print_section("Competitor Availability")
    print_table(["Framework", "Status", "Notes"], _availability_rows(report["competitor_availability"]))

    headers = ["Framework", "Pass", "Shape", "Status", "Median ms", "P95 ms", "GB/s", "Peak %", "Notes"]
    for sweep, title in SWEEP_TITLES.items():
        print_section(title)
        print_table(headers, _timing_rows(report["results"], sweep))

    print_section("VRAM")
    print_table(["Sweep", "Framework", "Pass", "Shape", "Status", "Peak"], _vram_rows(report["results"]))

    print_section("Profiling")
    print_table(["Sweep", "Framework", "Pass", "Shape", "AI", "GB/s", "Peak %", "Roofline"], _profiling_rows(report["results"]))

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
                    ["Casting Mode", settings["casting_mode"]],
                    ["Offset", settings["offset"]],
                    ["Eps", settings["eps"]],
                    ["Backward in_place", settings["backward_in_place"]],
                ],
            ),
            "## Competitor Availability",
            _md_table(["Framework", "Status", "Notes"], _availability_rows(report["competitor_availability"])),
        ]
    )

    headers = ["Framework", "Pass", "Shape", "Status", "Median ms", "P95 ms", "GB/s", "Peak %", "Notes"]
    for sweep, title in SWEEP_TITLES.items():
        parts.extend([f"## {title}", _md_table(headers, _timing_rows(report["results"], sweep))])

    parts.extend(
        [
            "## VRAM",
            _md_table(["Sweep", "Framework", "Pass", "Shape", "Status", "Peak"], _vram_rows(report["results"])),
            "## Profiling",
            _md_table(["Sweep", "Framework", "Pass", "Shape", "AI", "GB/s", "Peak %", "Roofline"], _profiling_rows(report["results"])),
            "## Warnings",
            _md_table(["Warning"], [[warning] for warning in report["warnings"]]),
        ]
    )
    return "\n".join(parts)

