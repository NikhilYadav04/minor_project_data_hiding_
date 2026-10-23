from pathlib import Path

import cv2
import numpy as np


def test_bundled_image_set_is_present_and_valid():
    root = Path(__file__).parents[1] / "data" / "images"
    names = ["airplane", "baboon", "boat", "couple", "elaine", "man", "peppers", "sailboat"]
    for name in names:
        image = cv2.imread(str(root / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        assert image is not None
        assert image.shape == (512, 512)
        assert image.dtype == np.uint8
