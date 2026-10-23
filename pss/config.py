GF_POLY = 0x11D
R = 3
N = 4
HEADER_BYTES = 16
MIN_HEADER_BYTES = 16


def validate_image_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2:
        raise ValueError("images must be two-dimensional grayscale arrays")
    m, n = shape
    if m <= 0 or n <= 0 or m % 2 or n % 4:
        raise ValueError("image dimensions must have even rows and columns divisible by four")
