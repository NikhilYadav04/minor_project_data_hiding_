import numpy as np
import pytest


@pytest.fixture
def image64():
    y, x = np.indices((64, 64))
    return ((x * 7 + y * 13 + (x * y) % 31) % 256).astype(np.uint8)
