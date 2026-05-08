"""Small dependency-free terminal table helpers."""
from __future__ import annotations


def print_section(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def format_float(value) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_verdict(value: str) -> str:
    return value


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    if not rows:
        print("(none)")
        return
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rendered))
        for i, header in enumerate(headers)
    ]
    header_line = " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers))
    sep_line = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(sep_line)
    for row in rendered:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
