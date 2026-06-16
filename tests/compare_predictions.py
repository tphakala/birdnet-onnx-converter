#!/usr/bin/env python3
"""Compare ONNX predictions against TFLite baseline.

This script compares bird detection predictions from an ONNX model against
a baseline CSV (typically from TFLite/BirdNET Analyzer) to verify that
the converted model produces accurate results.

Detections are compared by confidence value, not by membership in a
hard-thresholded set: a detection that merely straddles the --min-confidence
cutoff (present on one side, just under on the other) is tolerated when the two
models' confidences agree within --tolerance. A genuine divergence larger than
--tolerance still fails. This keeps the comparison stable across precision
variants (FP16/INT8) whose rounding can nudge a borderline detection across the
reporting threshold without changing the result in any meaningful way.

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


def parse_csv(path: str, floor: float = 0.0) -> list[Detection]:
    """Parse BirdNET Analyzer CSV format.

    Args:
        path: Path to CSV file
        floor: Drop rows below this confidence (default 0.0 keeps every row so
            near-cutoff values remain available for boundary comparison)

    Returns:
        List of Detection objects at or above the floor
    """
    detections = []
    with open(path, encoding="utf-8-sig") as f:  # Handle BOM in Windows-generated CSVs
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row["Confidence"])
            if conf >= floor:
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
    baseline: list[Detection],
    test: list[Detection],
    tolerance: float,
    min_confidence: float,
) -> tuple[bool, list[str], list[str]]:
    """Compare test detections against baseline by confidence value.

    A detection is "active" when its confidence is at or above min_confidence.
    For every detection active in either model, the two confidences are compared
    (a side that lacks the detection contributes its actual sub-threshold value,
    or 0.0 if absent entirely). A difference greater than tolerance is an error.
    A membership flip across the cutoff whose confidences still agree within
    tolerance is reported as a (non-failing) boundary note.

    Args:
        baseline: Baseline detections (ground truth), parsed with floor 0.0
        test: Test detections (from ONNX model), parsed with floor 0.0
        tolerance: Maximum allowed confidence difference
        min_confidence: Confidence threshold a detection must meet to be active

    Returns:
        Tuple of (passed, error messages, boundary notes)
    """
    errors: list[str] = []
    notes: list[str] = []

    # Round float keys to 2 decimal places to avoid floating-point comparison issues
    baseline_map = {(round(d.start, 2), round(d.end, 2), d.scientific_name): d for d in baseline}
    test_map = {(round(d.start, 2), round(d.end, 2), d.scientific_name): d for d in test}

    def conf(map_: dict, key: tuple) -> float:
        det = map_.get(key)
        return det.confidence if det is not None else 0.0

    active_baseline = {k for k, d in baseline_map.items() if d.confidence >= min_confidence}
    active_test = {k for k, d in test_map.items() if d.confidence >= min_confidence}

    for key in sorted(active_baseline | active_test):
        bc = conf(baseline_map, key)
        tc = conf(test_map, key)
        diff = abs(bc - tc)
        if diff <= tolerance:
            # If membership flipped across the cutoff, record a boundary note.
            if (key in active_baseline) != (key in active_test):
                name = (baseline_map.get(key) or test_map[key]).scientific_name
                notes.append(
                    f"Boundary detection at {(key[0], key[1])} for {name}: "
                    f"baseline={bc:.3f}, test={tc:.3f} (within tolerance of cutoff)"
                )
            continue

        segment = (key[0], key[1])
        name = (baseline_map.get(key) or test_map[key]).scientific_name
        if key not in active_test:
            errors.append(
                f"Missing detection at {segment}: {name} (baseline={bc:.2f}, test={tc:.2f})"
            )
        elif key not in active_baseline:
            errors.append(f"False positive at {segment}: {name} (baseline={bc:.2f}, test={tc:.2f})")
        else:
            errors.append(
                f"Confidence mismatch at {segment} for {name}: baseline={bc:.2f}, test={tc:.2f}"
            )

    return len(errors) == 0, sorted(errors), sorted(notes)


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

    # Parse the full CSVs so values just below the cutoff remain available for
    # boundary comparison; activeness is decided in compare() via min_confidence.
    baseline = parse_csv(args.baseline)
    test = parse_csv(args.test)

    passed, errors, notes = compare(baseline, test, args.tolerance, args.min_confidence)

    active_baseline = sum(1 for d in baseline if d.confidence >= args.min_confidence)
    active_test = sum(1 for d in test if d.confidence >= args.min_confidence)
    print(f"Baseline: {active_baseline} detections, Test: {active_test} detections")
    print(f"Tolerance: ±{args.tolerance:.2f}, Min confidence: {args.min_confidence:.2f}")

    for note in notes[:20]:
        print(f"  note: {note}")

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
