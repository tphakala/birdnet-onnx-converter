#!/usr/bin/env python3
"""Compare ONNX predictions against TFLite baseline.

This script compares bird detection predictions from an ONNX model against
a baseline CSV (typically from TFLite/BirdNET Analyzer) to verify that
the converted model produces accurate results.

Usage:
    python compare_predictions.py baseline.csv test.csv --tolerance 0.01
"""

import argparse
import csv
import sys
from dataclasses import dataclass


@dataclass
class Detection:
    """A single bird detection."""

    start: float
    end: float
    scientific_name: str
    common_name: str
    confidence: float


def parse_csv(path: str, min_confidence: float = 0.10) -> list[Detection]:
    """Parse BirdNET Analyzer CSV format, filtering by minimum confidence.

    Args:
        path: Path to CSV file
        min_confidence: Minimum confidence threshold (default: 0.10)

    Returns:
        List of Detection objects above the confidence threshold
    """
    detections = []
    with open(path, encoding="utf-8-sig") as f:  # Handle BOM in Windows-generated CSVs
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row["Confidence"])
            if conf >= min_confidence:
                detections.append(
                    Detection(
                        start=float(row["Start (s)"]),
                        end=float(row["End (s)"]),
                        scientific_name=row["Scientific name"],
                        common_name=row["Common name"],
                        confidence=conf,
                    )
                )
    return detections


def compare(
    baseline: list[Detection], test: list[Detection], tolerance: float
) -> tuple[bool, list[str]]:
    """Compare test detections against baseline.

    Args:
        baseline: List of baseline detections (ground truth)
        test: List of test detections (from ONNX model)
        tolerance: Maximum allowed confidence difference

    Returns:
        Tuple of (passed, list of error messages)
    """
    errors = []
    # Round float keys to 2 decimal places to avoid floating-point comparison issues
    baseline_map = {(round(d.start, 2), round(d.end, 2), d.scientific_name): d for d in baseline}
    test_map = {(round(d.start, 2), round(d.end, 2), d.scientific_name): d for d in test}

    baseline_keys = set(baseline_map.keys())
    test_keys = set(test_map.keys())

    # Check for missing detections (in baseline but not in test)
    for key in sorted(baseline_keys - test_keys):
        segment = (key[0], key[1])
        bd = baseline_map[key]
        errors.append(f"Missing detection at {segment}: {bd.scientific_name} ({bd.confidence:.2f})")

    # Check for false positives (in test but not in baseline)
    for key in sorted(test_keys - baseline_keys):
        segment = (key[0], key[1])
        td = test_map[key]
        errors.append(f"False positive at {segment}: {td.scientific_name} ({td.confidence:.2f})")

    # Check for confidence mismatch on common detections
    for key in sorted(baseline_keys & test_keys):
        bd = baseline_map[key]
        td = test_map[key]
        if abs(td.confidence - bd.confidence) > tolerance:
            segment = (key[0], key[1])
            errors.append(
                f"Confidence mismatch at {segment} for {bd.scientific_name}: "
                f"baseline={bd.confidence:.2f}, test={td.confidence:.2f}"
            )

    return len(errors) == 0, sorted(errors)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare ONNX predictions against TFLite baseline")
    parser.add_argument("baseline", help="Baseline predictions CSV (TFLite)")
    parser.add_argument("test", help="Test predictions CSV (ONNX)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Confidence tolerance (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.10,
        help="Minimum confidence threshold (default: 0.10)",
    )
    args = parser.parse_args()

    baseline = parse_csv(args.baseline, args.min_confidence)
    test = parse_csv(args.test, args.min_confidence)

    passed, errors = compare(baseline, test, args.tolerance)

    print(f"Baseline: {len(baseline)} detections, Test: {len(test)} detections")
    print(f"Tolerance: ±{args.tolerance:.2f}, Min confidence: {args.min_confidence:.2f}")

    if not passed:
        print(f"\nFAILED: {len(errors)} prediction mismatches")
        for e in errors[:20]:  # Show first 20
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 1

    print("\nPASSED: All detections verified within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
