from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HuffmanEncoded:
    bits: np.ndarray
    bit_length: int
    count: int
    canonical: bool


def _code_lengths(values: np.ndarray) -> dict[int, int]:
    symbols, counts = np.unique(values.astype(np.int32), return_counts=True)
    if len(symbols) == 1:
        return {int(symbols[0]): 1}
    heap = []
    serial = 0
    for symbol, count in zip(symbols.tolist(), counts.tolist()):
        heapq.heappush(heap, (int(count), int(symbol), serial, int(symbol)))
        serial += 1
    while len(heap) > 1:
        weight_a, min_a, _, left = heapq.heappop(heap)
        weight_b, min_b, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (weight_a + weight_b, min(min_a, min_b), serial, (left, right)))
        serial += 1
    root = heap[0][3]
    lengths: dict[int, int] = {}

    def visit(node, depth: int):
        if isinstance(node, int):
            lengths[node] = max(1, depth)
            return
        visit(node[0], depth + 1)
        visit(node[1], depth + 1)

    visit(root, 0)
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    code = 0
    previous_length = 0
    result = {}
    for symbol, length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= length - previous_length
        result[symbol] = (code, length)
        code += 1
        previous_length = length
    return result


def _tree_codes(lengths: dict[int, int], values: np.ndarray) -> dict[int, tuple[int, int]]:
    # Reconstruct a deterministic tree by using the same tie-breaking procedure and collect bits.
    symbols, counts = np.unique(values.astype(np.int32), return_counts=True)
    if len(symbols) == 1:
        return {int(symbols[0]): (0, 1)}
    heap = []
    serial = 0
    for symbol, count in zip(symbols.tolist(), counts.tolist()):
        heapq.heappush(heap, (int(count), int(symbol), serial, int(symbol)))
        serial += 1
    while len(heap) > 1:
        weight_a, min_a, _, left = heapq.heappop(heap)
        weight_b, min_b, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (weight_a + weight_b, min(min_a, min_b), serial, (left, right)))
        serial += 1
    codes = {}

    def visit(node, code: int, depth: int):
        if isinstance(node, int):
            codes[node] = (code, max(1, depth))
        else:
            visit(node[0], code << 1, depth + 1)
            visit(node[1], (code << 1) | 1, depth + 1)

    visit(heap[0][3], 0, 0)
    return codes


def _append_bits(target: list[int], value: int, width: int) -> None:
    target.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def compress_residuals(values, canonical: bool = False) -> HuffmanEncoded:
    residuals = np.asarray(values, dtype=np.int32).reshape(-1)
    if np.any(residuals < -255) or np.any(residuals > 255):
        raise ValueError("residual symbols must be in [-255, 255]")
    lengths = _code_lengths(residuals)
    codes = _canonical_codes(lengths) if canonical else _tree_codes(lengths, residuals)
    max_len = max(lengths.values())
    width = max(1, int(np.ceil(np.log2(max_len + 1))))
    if width > 127:
        raise ValueError("Huffman code length does not fit the format")
    bits: list[int] = []
    _append_bits(bits, width | (0x80 if canonical else 0), 8)
    for symbol in range(-255, 256):
        length = lengths.get(symbol, 0)
        _append_bits(bits, length, width)
        if length:
            _append_bits(bits, codes[symbol][0], length)
    for symbol in residuals.tolist():
        code, length = codes[int(symbol)]
        _append_bits(bits, code, length)
    result = np.asarray(bits, dtype=np.uint8)
    return HuffmanEncoded(result, len(result), len(residuals), canonical)


def pad_bits(bits: np.ndarray, multiple: int = 16) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    padding = (-len(values)) % multiple
    if padding:
        values = np.concatenate((values, np.zeros(padding, dtype=np.uint8)))
    return values


def pack_bits(bits: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big")


def _read_bits(bits: np.ndarray, position: int, width: int) -> tuple[int, int]:
    end = position + width
    if end > len(bits):
        raise ValueError("truncated Huffman stream")
    value = 0
    for bit in bits[position:end].tolist():
        value = (value << 1) | int(bit)
    return value, end


def decompress_residuals(bits, count: int) -> np.ndarray:
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1)
    width_marker, position = _read_bits(stream, 0, 8)
    canonical = bool(width_marker & 0x80)
    width = width_marker & 0x7F
    if width < 1:
        raise ValueError("invalid Huffman width")
    lengths: dict[int, int] = {}
    stored_codes: dict[tuple[int, int], int] = {}
    for symbol in range(-255, 256):
        length, position = _read_bits(stream, position, width)
        lengths[symbol] = length
        if length:
            if canonical:
                continue
            code, position = _read_bits(stream, position, length)
            stored_codes[(code, length)] = symbol
    if canonical:
        stored_codes = {(code, length): symbol for symbol, (code, length) in _canonical_codes({s: l for s, l in lengths.items() if l}).items()}
    decoded = []
    code = 0
    length = 0
    while len(decoded) < count:
        if position >= len(stream):
            raise ValueError("truncated Huffman payload")
        code = (code << 1) | int(stream[position])
        position += 1
        length += 1
        symbol = stored_codes.get((code, length))
        if symbol is not None:
            decoded.append(symbol)
            code = 0
            length = 0
        elif length > max(lengths.values()):
            raise ValueError("invalid Huffman code")
    return np.asarray(decoded, dtype=np.int32)
