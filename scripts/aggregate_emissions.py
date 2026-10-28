#!/usr/bin/env python3
"""Aggregate CodeCarbon emissions across MCP tools.

Reads all CSV files in the shared emissions directory and produces a summary
by project name and overall totals.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable


def _safe_float(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_csv_files(input_dir: Path) -> Iterable[Path]:
    patterns = ["*.csv", "*.csv_*", "*.bak"]
    seen: set[Path] = set()
    for pattern in patterns:
        for path in input_dir.glob(pattern):
            if path.name == "aggregated_emissions.csv":
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def _iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            yield row


def aggregate_emissions(input_dir: Path) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "emissions": 0.0,
        "energy": 0.0,
        "cpu_energy": 0.0,
        "gpu_energy": 0.0,
        "ram_energy": 0.0,
        "records": 0.0,
    })

    for csv_path in _iter_csv_files(input_dir):
        for row in _iter_rows(csv_path):
            project_name = row.get("project_name") or "unknown"
            summary[project_name]["emissions"] += _safe_float(row.get("emissions"))
            summary[project_name]["energy"] += _safe_float(row.get("energy_consumed"))
            # Component breakdown lets us split the orchestrator request into
            # Layer 1 (gpu_energy ≈ LLM) vs Layer 3 (cpu+ram ≈ overhead).
            summary[project_name]["cpu_energy"] += _safe_float(row.get("cpu_energy"))
            summary[project_name]["gpu_energy"] += _safe_float(row.get("gpu_energy"))
            summary[project_name]["ram_energy"] += _safe_float(row.get("ram_energy"))
            summary[project_name]["records"] += 1

    return summary


def write_aggregated_csv(output_file: Path, summary: Dict[str, Dict[str, float]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "project_name", "total_emissions_kg", "total_energy_kwh",
            "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh", "records",
        ])
        for project_name in sorted(summary.keys()):
            data = summary[project_name]
            writer.writerow([
                project_name,
                f"{data['emissions']:.6f}",
                f"{data['energy']:.6f}",
                f"{data['cpu_energy']:.6f}",
                f"{data['gpu_energy']:.6f}",
                f"{data['ram_energy']:.6f}",
                int(data["records"]),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CodeCarbon emissions logs.")
    parser.add_argument(
        "--input-dir",
        default="logs/emissions",
        help="Directory containing CodeCarbon emissions CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default="logs/emissions/aggregated_emissions.csv",
        help="Path for aggregated summary CSV.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    summary = aggregate_emissions(input_dir)
    if not summary:
        raise SystemExit("No emissions CSV files found to aggregate.")

    output_file = Path(args.output_file)
    write_aggregated_csv(output_file, summary)

    total_emissions = sum(data["emissions"] for data in summary.values())
    total_energy = sum(data["energy"] for data in summary.values())
    total_records = int(sum(data["records"] for data in summary.values()))

    print("Emissions summary (kg CO2e):")
    for project_name in sorted(summary.keys()):
        data = summary[project_name]
        line = (
            f"- {project_name}: {data['emissions']:.6f} kg, "
            f"{data['energy']:.6f} kWh across {int(data['records'])} records"
        )
        # Show the component split when present (orchestrator_request → L1 vs L3).
        if data["gpu_energy"] or data["cpu_energy"] or data["ram_energy"]:
            line += (
                f"  [gpu={data['gpu_energy']:.6f} kWh (L1 LLM) | "
                f"cpu+ram={data['cpu_energy'] + data['ram_energy']:.6f} kWh (L3 overhead)]"
            )
        print(line)
    print(
        f"Total: {total_emissions:.6f} kg, {total_energy:.6f} kWh across {total_records} records"
    )
    print(f"Aggregated CSV written to: {output_file}")


if __name__ == "__main__":
    main()
