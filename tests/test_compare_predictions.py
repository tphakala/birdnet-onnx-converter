"""Tests for the prediction-comparison logic (boundary-aware confidence diff)."""

from compare_predictions import Detection, compare


def _det(conf: float, name: str = "Tetrastes bonasia", start: float = 0.0) -> Detection:
    return Detection(
        start=start, end=start + 3.0, scientific_name=name, common_name="x", confidence=conf
    )


def test_identical_predictions_pass():
    base = [_det(0.30), _det(0.50, "Ficedula parva")]
    passed, errors, _notes = compare(base, list(base), tolerance=0.01, min_confidence=0.10)
    assert passed and not errors


def test_boundary_flicker_within_tolerance_passes():
    """A detection at 0.15 in baseline and just under in test must not fail when
    the values agree within tolerance, even though it straddles the cutoff."""
    base = [_det(0.150)]
    test = [_det(0.149)]
    passed, errors, notes = compare(base, test, tolerance=0.05, min_confidence=0.15)
    assert passed, errors
    assert not errors
    assert len(notes) == 1  # recorded as a boundary note, not an error


def test_real_missing_detection_fails():
    """A baseline detection well above the cutoff that is absent from test
    (confidence collapses) must fail."""
    base = [_det(0.30)]
    test: list[Detection] = []  # absent entirely -> test confidence 0.0
    passed, errors, _notes = compare(base, test, tolerance=0.05, min_confidence=0.15)
    assert not passed
    assert any("Missing detection" in e for e in errors)


def test_confidence_mismatch_above_tolerance_fails():
    base = [_det(0.40)]
    test = [_det(0.20)]
    passed, errors, _notes = compare(base, test, tolerance=0.05, min_confidence=0.15)
    assert not passed
    assert any("Confidence mismatch" in e for e in errors)


def test_false_positive_far_from_cutoff_fails():
    """A detection only the test model reports, well above the cutoff, fails."""
    base: list[Detection] = []
    test = [_det(0.40)]
    passed, errors, _notes = compare(base, test, tolerance=0.05, min_confidence=0.15)
    assert not passed
    assert any("False positive" in e for e in errors)


def test_boundary_flip_just_over_tolerance_fails():
    """Straddling the cutoff is only tolerated within tolerance; a larger gap fails."""
    base = [_det(0.15)]
    test = [_det(0.05)]  # diff 0.10 > tolerance 0.05
    passed, errors, _notes = compare(base, test, tolerance=0.05, min_confidence=0.15)
    assert not passed
    assert errors
