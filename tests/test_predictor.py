import numpy as np

from pss.predictor import get_lr, predict, rebuild_residual, residual, residual_sequence


def test_predictors_preserve_lr_and_constant():
    low = np.full((32, 32), 128, dtype=np.uint8)
    for method in ("bilinear", "edge_directed", "med"):
        high = predict(low, method)
        assert high.dtype == np.uint8
        assert np.array_equal(high[0::2, 0::2], low)
        assert np.all(residual(high, high) == 0)


def test_residual_round_trip(image64):
    low = get_lr(image64)
    high = predict(low, "edge_directed")
    errors = residual(image64, high)
    sequence = residual_sequence(errors)
    assert np.array_equal(rebuild_residual(sequence, image64.shape), errors)


def test_med_is_deterministic_and_preserves_lr(image64):
    low = get_lr(image64)
    first = predict(low, "med")
    second = predict(low, "med")
    assert np.array_equal(first, second)
    assert np.array_equal(first[0::2, 0::2], low)
