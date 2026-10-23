# Progressive Secret Image Sharing with Reversible Data Hiding in Encrypted Images
### A simplified reproduction of He et al. (IEEE IoT Journal, 2026) — Minor Project Specification

| | |
|---|---|
| **Base paper** | M. He, X. Xie, X. Wang, L. Zhang, "Progressive Secret Sharing Through the Integration of Deep Learning and Reversible Data Hiding in Encrypted Images," *IEEE Internet of Things Journal*, vol. 13, no. 3, pp. 4564–4577, 1 Feb 2026. DOI 10.1109/JIOT.2025.3635284 |
| **Scheme name in paper** | PSS-DRDHEI |
| **Team / time** | 2 people, 4 days |
| **Language / stack** | Python 3.10+, NumPy, OpenCV, scikit-image, pandas, Matplotlib, pytest, Streamlit (UI). **No deep-learning frameworks.** |
| **Target configuration** | (r, n) = (3, 4) threshold, 512×512 grayscale 8-bit images |
| **Document purpose** | (1) project proposal, and (2) complete implementation spec to hand to Claude Code |

---

## 0. How to use this document with Claude Code

1. Put this file in the repo root as `SPEC.md` (or `CLAUDE.md`).
2. Put the 8 test images in `data/images/` (see §4.2).
3. Start Claude Code and paste the kickoff prompt below.

### Kickoff prompt (copy-paste)

> Read `SPEC.md` completely before writing any code. We are building a simplified reproduction of the PSS-DRDHEI scheme described there. Follow §5 (algorithm), §6 (module API) and §7 (tests) exactly. Work in the milestone order of §11 and do not start a milestone until the previous milestone's tests pass. Rules: no PyTorch/TensorFlow; all randomness comes from seeded `numpy.random.Generator` objects derived from one master seed; every function that changes pixels or bytes must have a round-trip test; after each milestone run `pytest -q` and show me the result. When the spec is ambiguous or you must deviate, stop and record the decision in `DEVIATIONS.md` instead of silently choosing. Do not claim any result you have not produced by running code.

### Working rules for Claude Code
- Implement milestone by milestone (§11). Run tests after each.
- Vectorise with NumPy. No per-pixel Python loops in GF(2⁸) sharing/recovery (a 512×512 image has 131 072 pixel pairs per share; a Python loop is acceptable only in Huffman decoding).
- Use `uint8` for pixels/bytes, `int32` for any arithmetic that can go negative or exceed 255.
- Every experiment writes a CSV to `results/` and figures to `figures/`, and logs its seed and parameters.

---

## 1. Project summary

Cloud and IoT systems store private images. This project implements a **progressive (r, n)-threshold secret image sharing scheme** in which an image owner splits one grayscale image into *n* noise-like shares. Different numbers of shares and keys give different outputs:

| What the receiver holds | What can be recovered |
|---|---|
| fewer than *r* shares | nothing useful |
| any *r* shares + LR key | **Stage 1:** low-resolution (LR) image, size M/2 × N/2 |
| all *n* shares + LR key + code key | **Stage 2:** the *masked* image (full size, sensitive region replaced by fake content) |
| all *n* shares + all keys + MSA key | **Stage 3:** the exact **original** image (lossless) |

The "most sensitive area" (MSA) is a rectangle (for example a face, a tank, a lesion) that is cut out, protected separately, and replaced by inpainted content in the masked image. The shares additionally have **spare room** in which a third party (a "data hider", e.g. a cloud server) can embed extra data (signatures, timestamps) without any key to the image, and that data can be extracted without recovering the image.

**Project goal:** reproduce the full pipeline end to end, verify lossless recovery at every stage, measure embedding rate and security metrics on 8 standard images, and honestly compare with the paper's reported numbers.

---

## 2. Scope: what is reproduced, simplified, or dropped

| Paper component | In this project | Reason |
|---|---|---|
| Rectangular MSA peeling | **Reproduced** | Core feature |
| GICA neural inpainting | **Replaced** by `cv2.inpaint` (Telea) | The receiver never re-runs inpainting: Stage 2 recovers the masked image from the residual, so any fill method is valid. Only visual quality differs. |
| 2×2 down-sampling to LR image | **Reproduced** | Core |
| RCAN super-resolution predictor | **Replaced** by deterministic integer predictors (§5.5) | Training/running RCAN is infeasible in 4 days. Predictor must be bit-exact on sender and receiver. |
| Prediction residual + Huffman coding with matching rules | **Reproduced** | Core |
| Shamir sharing over GF(2⁸): (r,n) for LR, (n,n) for code and MSA | **Reproduced** | Core |
| 1×2 pixel-pair packing (c and a₁ carry data) | **Reproduced** | Core |
| Expansion to 2×4 blocks, embedding in redundant pixels | **Reproduced** | Core |
| Data hider with separate extraction | **Reproduced** | Core |
| 3-stage progressive recovery | **Reproduced** | Core |
| Security metrics (entropy, correlation, NPCR, UACI, histogram, key space) | **Reproduced** | Paper §IV-C |
| Predictor ablation (paper: RCAN vs MED) | **Adapted**: our predictors vs each other | Paper Figs. 9–10 |
| BOSSbase 10 000 images | **Dropped** (optional: small subset if available) | No training needed |
| Comparison with 5 other schemes | **Quoted from paper's Table III**, clearly labelled as quoted | Not re-implemented |
| GUI | **Reproduced as a small Streamlit demo** (§6.1) | Required by the team; built last, on top of the tested pipeline |

> **Honesty rule:** the report must say "simplified reproduction". Embedding rates will differ from the paper because the predictor differs. Never claim to beat the paper or other schemes.

### 2.1 Priority tiers (what to build, in order)

| Tier | Contents | Rule |
|---|---|---|
| **Tier 1 — must work** | Full pipeline with 3-stage lossless recovery, MSA, data hider, both predictors, security metrics (E1–E6), Streamlit UI with Owner / Data hider / Receiver tabs | Done by end of Day 3 |
| **Tier 2 — makes the project stand out** | E7 MSA size sweep, E8 predictor ablation (3 predictors incl. cv2 bicubic), E9 ramp-leakage measurement (finding D3), E10 data-hider capacity test, canonical-Huffman header comparison, UI Metrics tab | Start only after Tier 1 passes; Day 3–4 |
| **Tier 3 — stretch** | Tiny CNN predictor used only for the residual-entropy comparison (§6.2) | Only if everything else is finished; expect to drop it |

What impresses examiners: exact recovery proven on every share combination, a measured security analysis, discrepancies found in the published paper (D1–D3, D7) with evidence, a predictor ablation, and a working demo. This is depth, not volume.

---

## 3. Facts from the paper and discrepancies found while analysing it

### 3.1 Facts used
- Image M×N pixels, 8-bit grayscale. All three payloads (LR image, compressed residual code **C**, MSA **A**) are shared byte-wise over GF(2⁸).
- LR image **L** = first pixel of every non-overlapping 2×2 block of the masked image **M** (size M/2 × N/2).
- HR prediction **H** = predictor(**L**), then overwrite H's first pixel of each 2×2 block with L (paper Eq. 7). Residual **E = M − H**, range [−255, 255], first value in every block is 0.
- Residual (three values per block) is Huffman-coded. Compressed sequence **C = matching rules ‖ encoded residual** (Eqs. 9–11).
- Sharing uses only the constant term *c* and first coefficient *a₁* of a degree-(r−1) polynomial for data; remaining coefficients are random.
- LR: (r, n)-threshold on 1×2 pixel pairs → each share is M/2 × N/4.
- C and A: (n, n)-threshold. C is cut into 16-bit pairs; A is stream-cipher-encrypted, then paired.
- Each M/2×N/4 shared image is **expanded** to M×N by replacing each pixel with a 2×4 block: first pixel = shared pixel, other 7 = random (the "redundant pixels").
- Shares of C then shares of A overwrite redundant pixels; coordinates of the boundary are stored at the front. Remaining redundancy is free for the data hider.
- Embedding rate: **r = (7·M·N − ℓ(S_C)) / (M·N)** bits per pixel, where ℓ(S_C) is the bit length of one share of the compressed code.
- Paper's quoted results (Table III, "Proposed", bpp): Airplane 5.4867, Baboon 5.1466, Boat 5.0487, Couple 5.1721, Elaine 5.1708, Man 5.4330, Peppers 5.0868, Sailboat 5.2857, **average 5.2288**. Best competitor ("[32] full size") average 4.3072. Ideal NPCR = 99.6094 %, ideal UACI = 33.4635 %.

### 3.2 Discrepancies / issues (log all of these in the report)

| # | Issue | How we handle it |
|---|---|---|
| D1 | **Irreducible polynomial.** Eq. (3) prints p(x)=x⁸+x⁴+x³+x+1 (0x11B). But the paper's own worked example (Fig. 5) is only reproduced by x⁸+x⁴+x³+x²+1 (**0x11D**). Verified: with 0x11D, pair (73,76), a₂=106, K=110 gives shares 236, 202, 53, 19; pair (77,78), a₂=184, K=6 gives 97, 151, 38, 208 (exactly Fig. 5). With 0x11B the values differ. | Default polynomial = **0x11D** (config constant `GF_POLY`), so the paper's example passes as a test. Both polynomials are irreducible; either works functionally. |
| D2 | **Evaluation points.** The paper says keys K ∈ [0,255] and uses x = K, K+1, … K+n−1. x = 0 makes a share equal to the secret constant c, and K+n−1 may exceed 255. | We draw K ∈ [1, 256−n] so all x_i ∈ [1,255], distinct, non-zero. |
| D3 | **Ramp-scheme leakage.** Two secret symbols (c, a₁) are carried by a degree-(r−1) polynomial with only r−2 random coefficients. With r−1 shares and the evaluation points known, the number of candidate (c, a₁) pairs shrinks from 65 536 to 256. The paper states r−1 shares reveal nothing. | The evaluation points depend on the secret key K, so an attacker without K cannot exploit this. Report it as an observation in the report; do not change the scheme. |
| D4 | Table III average (5.2288 bpp) differs from Table IV / Fig. 12 value for the proposed scheme (5.6007 bpp, which appears to be the median over 2 000 images). Also the text says "ten test images" while only eight are shown. | We use the 8 shown images and compare against Table III. |
| D5 | RCAN configuration is stated twice with different values (5 groups/5 blocks in §II, 6 groups/10 blocks in §IV). | Not applicable (we do not use RCAN); mention in report. |
| D6 | Equations (12)–(14) use "a_k−1" notation inconsistent with the polynomial degree r−1. | We implement the polynomial as in Eq. (2). |
| D7 | Lossless recovery with a neural predictor requires **bit-identical** network output on sender and receiver. The paper does not discuss floating-point nondeterminism across devices. | Our integer predictors avoid the issue. Mention as a practical limitation of the original design. |

---

## 4. Environment and repository

### 4.1 Environment
```
python >= 3.10
numpy, opencv-python, scikit-image, pandas, matplotlib, pytest, streamlit
```
No GPU. No internet needed after install.

### 4.2 Test data
Eight 512×512 8-bit grayscale PNG files in `data/images/`, named exactly:
`airplane.png, baboon.png, boat.png, couple.png, elaine.png, man.png, peppers.png, sailboat.png`
(the same set as the paper's Fig. 7). If an image is not 512×512 grayscale, a helper `load_gray512()` converts it (grayscale, resize with INTER_AREA). Also keep a 64×64 crop of each for fast unit tests. Images are **not** committed to git if their license forbids it; document their source in `data/README.md`.

### 4.3 Layout
```
pss/
  __init__.py
  config.py        # GF_POLY, R=3, N=4, header constants
  gf256.py         # field arithmetic
  keys.py          # KeyBundle from master seed
  sharing.py       # (k,n) pair sharing and recovery over GF(256)
  msa.py           # rectangle cut, pack/unpack, inpaint, stream cipher
  predictor.py     # bilinear, edge_directed, (ablation) cv2_bicubic
  huffman.py       # build code, matching rules, encode/decode residual
  embed.py         # expand, embed payload, extract, data hider
  pipeline.py      # owner_encode(), recover_stage1/2/3()
  metrics.py       # psnr, ssim, entropy, correlation, npcr, uaci, bpp
  cli.py           # demo command
app.py               # Streamlit demo UI (section 6.1)
tests/
  test_gf256.py  test_sharing.py  test_predictor.py  test_huffman.py
  test_embed.py  test_pipeline.py  test_negative.py
experiments/
  run_all.py       # E1–E10, writes results/*.csv and figures/*.png
notebooks/demo.ipynb
results/  figures/  data/images/
SPEC.md  DEVIATIONS.md  README.md
```

---

## 5. Algorithm specification

### 5.0 Notation and conventions
- Image `img`: `uint8` array shape (M, N). Requirement: **M even, N divisible by 4** (512×512 satisfies this). Raise `ValueError` otherwise.
- (r, n) = (3, 4). Share index i ∈ {0, …, n−1}. Evaluation point of share i for a pair with key K is **x_i = K + i** (K ∈ [1, 256−n]).
- Bit arrays: `uint8` of 0/1. Byte arrays: `uint8`. Use `np.packbits` / `np.unpackbits` with big-endian bit order everywhere.
- Integers in headers: big-endian `uint32`.
- All randomness: `rng = np.random.default_rng(np.random.SeedSequence(master_seed).spawn(k)[j])`, one stream per purpose (see §5.2).

### 5.1 GF(2⁸) arithmetic (`gf256.py`)
- Addition and subtraction = bitwise XOR.
- Reduction polynomial `GF_POLY = 0x11D` (see D1). Build `EXP` (length 510) and `LOG` (length 256) tables by finding a primitive element programmatically (do not hard-code the generator).
- Vectorised functions on `uint8` arrays: `mul(a,b)`, `inv(a)` (error if a=0), `div(a,b)`, `pow(a,e)`.
- Self-checks: `mul(a, inv(a)) == 1` for all a≠0; commutativity and distributivity on random samples.

### 5.2 Keys (`keys.py`)
`KeyBundle(master_seed)` derives independent generators/arrays for:

| Name | Purpose | Shape/size |
|---|---|---|
| `K_L` | evaluation key per LR pixel pair | MN/8 values in [1, 256−n] |
| `K_C` | evaluation key per 16-bit pair of C | ⌈ℓ(C)/16⌉ values |
| `K_A` | evaluation key per MSA byte pair | ⌈len(A)/2⌉ values |
| `K_sc` | stream-cipher seed for the MSA (keystream bytes) | one integer seed |
| `K_dh` | data-hiding key (keystream for additional data) | one integer seed |
| `rng_coef_L/C/A` | random polynomial coefficients | internal, not a key |
| `rng_fill` | random fill of redundant pixels | internal |

Key arrays are generated lazily with the length needed. Because arrays are seed-derived, a "key" in this project = `(master_seed, purpose)`. State in the report that production key management is out of scope.

### 5.3 Pair sharing over GF(2⁸) (`sharing.py`)

```python
def share_pairs(pairs, K, k, n, rng) -> np.ndarray:
    """pairs: (P,2) uint8 -> (c, a1). K: (P,) uint8 in [1,256-n].
    Polynomial per pair: f(x) = c ^ a1*x ^ a2*x^2 ^ ... ^ a_{k-1}*x^{k-1}
    with a2..a_{k-1} fresh random uint8 per pair.
    Returns shares: (n, P) uint8, shares[i] = f(K + i)."""

def recover_pairs(shares, idx, K, k) -> np.ndarray:
    """shares: (k,P) uint8, idx: list of the k share indices used,
    K: (P,) keys. Returns pairs (P,2) = (c, a1)."""
```

**Recovery algorithm** (per pair, vectorised across pairs): the points are x_j = K + idx_j. For each used share j, let ℓ_j(x) = ∏_{m≠j}(x ⊕ x_m) / d_j with d_j = ∏_{m≠j}(x_j ⊕ x_m). Only the two lowest coefficients of the numerator are needed. Start `p0=1, p1=0`; for each m≠j update `(p0, p1) ← (p0·x_m, p0 ⊕ p1·x_m)`. Then
`c = ⊕_j y_j · p0_j / d_j` and `a1 = ⊕_j y_j · p1_j / d_j`.
(Truncating to two coefficients is valid because low-order coefficients depend only on low-order ones.)

Use:
- LR image: `k = r = 3`, `n = 4`, polynomial degree 2, **one** random coefficient a₂.
- Compressed code C: `k = n = 4`, degree 3, random a₂, a₃.
- MSA A: `k = n = 4`, same as C.

### 5.4 MSA preprocessing (`msa.py`)
Input: `rect = (top, left, height, width)`, half-open region rows `[top, top+height)`, cols `[left, left+width)`. Must lie inside the image and have area ≥ 1.

1. `A = img[rect]` (the sensitive pixels).
2. Build masked image: `mask = 255` inside rect else 0; `masked = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)`. Pixels outside the rectangle must be unchanged (assert it).
3. Pack: `A_bytes = header(8 bytes: top,left,bottom,right as 4 big-endian uint16, exclusive bottom/right) ‖ A.flatten()`. If the length is odd, append one zero byte. (Replaces the paper's "2·log₂(MN) coordinate bits".)
4. Stream-cipher encrypt: `A_enc = A_bytes XOR keystream(K_sc, len)` using `np.random.default_rng(K_sc).integers(0,256,len)`.
5. Pair as `A_enc.reshape(-1,2)`.

If the image has **no MSA**, skip all of this (`masked = img`, `LEN_SA = 0`).

Stage-3 reverse: decrypt with `K_sc`, parse header, **validate** the rectangle (inside image, consistent with byte count) and raise `InvalidKeyError` if not, then `out = masked.copy(); out[rect] = A`.

### 5.5 Intermediary images and predictors (`predictor.py`)

- `L = masked[0::2, 0::2]` (shape M/2 × N/2).
- `H = predict(L, method)` returns `uint8` (M, N) with `H[0::2,0::2] == L` (paper Eq. 7).
- Residual `E = masked.astype(int32) − H.astype(int32)`, range [−255, 255]. Assert `E[0::2,0::2] == 0`.
- Residual sequence **Ē**: for each 2×2 block in raster block order, the three values at positions (0,1), (1,0), (1,1). Length 3MN/4. Implementation: view as `(M/2, 2, N/2, 2)`, transpose to `(M/2, N/2, 2, 2)`, reshape `(-1, 4)`, take columns 1..3 and flatten.

**All predictors must use integer arithmetic only** (shifts, integer division) so sender and receiver agree bit for bit. Edge handling: replicate the last row/column of L.

Let `a=L[i,j]`, `b=L[i,j+1]` (right), `c=L[i+1,j]` (down), `e=L[i+1,j+1]` (diagonal), with replication at borders.

**`bilinear`**
- `H[2i,2j] = a`
- `H[2i,2j+1] = (a + b + 1) >> 1`
- `H[2i+1,2j] = (a + c + 1) >> 1`
- `H[2i+1,2j+1] = (a + b + c + e + 2) >> 2`

**`edge_directed`** (three passes)
1. Centres: `H[2i+1,2j+1] = (a+e+1)>>1 if |a−e| ≤ |b−c| else (b+c+1)>>1`.
2. Horizontal midpoints `H[2i,2j+1]`: let `U = H[2i−1,2j+1]`, `D = H[2i+1,2j+1]` (for i=0 use D for U). `= (a+b+1)>>1 if |a−b| ≤ |U−D| else (U+D+1)>>1`.
3. Vertical midpoints `H[2i+1,2j]`: let `Lf = H[2i+1,2j−1]`, `Rt = H[2i+1,2j+1]` (for j=0 use Rt for Lf). `= (a+c+1)>>1 if |a−c| ≤ |Lf−Rt| else (Lf+Rt+1)>>1`.

**`cv2_bicubic`** (Tier 2, **ablation only**): `cv2.resize(L, (N,M), INTER_CUBIC)` then overwrite even-even positions with L. Not used for the receiver-side guarantee because cv2 versions may differ; used only to compare residual statistics in E8.

Tests: shape, dtype, `H[0::2,0::2]==L`, a constant image predicts perfectly (E≡0), determinism (same output twice).

### 5.6 Residual compression (`huffman.py`)

1. Histogram of Ē over symbols −255…255 (511 symbols).
2. Build a Huffman code (`heapq`). Degenerate case: if only one symbol occurs, give it length 1, code `0`. Use a deterministic tie-break (by symbol) so the code is reproducible.
3. Encode Ē → bitstream **Ẽ**.
4. **Matching rules 𝓜** (paper Eq. 10), for h = −255 … 255 in order: write `ℓ₂(D_h)` using `w` bits, then the `ℓ(D_h)` code bits. Symbols that never occur have length 0 and no code bits. `w = max(1, ceil(log2(max_len + 1)))`. (Slightly different from the paper's `ceil(log2 max ℓ)` because we need to represent length 0; log this in `DEVIATIONS.md`.)
5. `C = [w as 8 bits] ‖ 𝓜 ‖ Ẽ`. Record `BITLEN_C = len(C)` (header field). Pad C with zero bits to a multiple of 16.
6. Decoder: read `w`, read the 511 (length, code) entries, build a dict, then decode exactly `3MN/4` symbols. Must reproduce Ē exactly.
7. Tier 2 flag `canonical=True`: store only the code lengths and generate canonical codes (much smaller header). Default **False** (paper-faithful). Report the bpp difference in E8.

C → 16-bit pairs: `C_bytes = packbits(C_padded)`, `pairs = C_bytes.reshape(-1,2)`.

### 5.7 Sharing the three payloads (`pipeline.py`)

| Payload | Pairs | Threshold | Output per share |
|---|---|---|---|
| LR image | `L.reshape(-1,2)` → MN/8 pairs (c = left pixel, a₁ = right pixel) | (r, n) = (3, 4) | MN/8 bytes → shape (M/2, N/4) |
| Compressed code C | `C_bytes.reshape(-1,2)` | (n, n) = (4, 4) | `S_C` : BITLEN/16 bytes (= half of padded C) |
| MSA | `A_enc.reshape(-1,2)` | (n, n) = (4, 4) | `S_A` : len(A_enc)/2 bytes |

### 5.8 Expansion and embedding (`embed.py`)

**Expansion.** For share i, the shared LR image `S_L^i` has shape (M/2, N/4). The expanded share `X_i` has shape (M, N): view as `(M/2, 2, N/4, 4)`, transpose to `(M/2, N/4, 2, 4)`, reshape `(-1, 8)`. Column 0 of each row (top-left pixel of the 2×4 block) = the shared pixel; columns 1…7 = random bytes from `rng_fill`. The **redundant stream** of a share is `blocks[:,1:].reshape(-1)`, length **7MN/8 bytes** (= 7MN bits).

**Embedded byte layout in the redundant stream** (identical structure in every share):

| Offset | Content |
|---|---|
| 0–3 | `LEN_SC` : bytes of S_C (uint32) |
| 4–7 | `LEN_SA` : bytes of S_A (uint32, 0 if no MSA) |
| 8–11 | `BITLEN_C` (uint32) |
| 12–15 | reserved = 0 |
| 16 … | `S_C` bytes |
| … | `S_A` bytes |
| `F = 16 + LEN_SC + LEN_SA` … end | **free room** (random fill from the owner) |

Assert capacity: `16 + LEN_SC + LEN_SA ≤ 7MN/8`, else raise `CapacityError` (tell the user the MSA is too large).

The result after the owner's embedding is the **final share** `S_i` (shape (M, N), uint8).

**Data hider** (no image keys needed). For share i with additional data `D_i` (bytes) and hiding key `K_dh,i`:
- Payload = `LEN (4 B) ‖ CRC32 (4 B) ‖ D_i`, all XOR-ed with `keystream(K_dh,i)`.
- Read the header to find `F`, overwrite the free room from offset `F` with this payload. Capacity: `7MN/8 − F − 8` bytes. Output is the **embedded share**.
- Extraction: read header → `F` → decrypt with `K_dh` → check CRC32 → return `D_i` (raise `ExtractionError` on CRC mismatch).
- Extraction does not touch recovery and recovery ignores the free room, so the embedded share recovers the image **identically** to the final share.

**Embedding-rate formulas** (report both):
- Paper formula: `bpp_paper = (7·M·N − 8·(LEN_SC)) / (M·N)` (+ subtract `8·LEN_SA` when an MSA is present).
- Practical free room: `bpp_free = 8·(7MN/8 − F − 8) / (M·N)` (includes our header and the 8-byte data-hider header). Always state the denominator (M·N original pixels).

### 5.9 Progressive recovery (`pipeline.py`)

**Stage 1 — LR image, any r = 3 shares + `K_L`**
1. For each share, take column 0 of its 2×4 blocks → shape (M/2, N/4).
2. `recover_pairs(shares, idx, K_L, k=3)` → `(MN/8, 2)` → reshape to `(M/2, N/2)` = **L**. Must equal the original L exactly.

**Stage 2 — masked image, all n = 4 shares + `K_L` + `K_C`**
1. Stage 1 using any 3 shares → L. `H = predict(L, method)`.
2. Read the header and `S_C` from each share. `recover_pairs(..., k=4, K_C)` → `C_bytes`; truncate bits to `BITLEN_C`.
3. Parse `w`, 𝓜, Ẽ; Huffman-decode to Ē; rebuild E (zeros at block-first positions).
4. `masked = H.astype(int32) + E`; assert in [0, 255]; cast to `uint8`. Must equal the owner's masked image exactly.

**Stage 3 — original image, Stage 2 + `K_A` + `K_sc`**
1. Read `S_A` from each share; `recover_pairs(..., k=4, K_A)`; decrypt with `K_sc`; parse and validate the rectangle; paste into masked image. Must equal the original image **exactly** (PSNR = ∞).

If there was no MSA (`LEN_SA = 0`), Stage 3 returns the masked image (which then equals the original).

---

## 6. Public API (`pipeline.py`)

```python
@dataclass
class OwnerOutput:
    final_shares: list[np.ndarray]       # n arrays (M,N) uint8
    expanded_shares: list[np.ndarray]    # before embedding S_C / S_A (for metrics)
    masked: np.ndarray                   # (M,N) uint8
    stats: dict                          # LEN_SC, LEN_SA, BITLEN_C, bpp_paper, bpp_free,
                                         # t_inpaint, t_predict, t_compress, t_share, t_embed

def owner_encode(img, rect, keys: KeyBundle, method="edge_directed", r=3, n=4) -> OwnerOutput
def hider_embed(final_share, data: bytes, k_dh: int) -> np.ndarray
def hider_extract(embedded_share, k_dh: int) -> bytes
def recover_stage1(shares, idx, keys, method) -> np.ndarray   # L
def recover_stage2(shares, keys, method) -> np.ndarray        # masked
def recover_stage3(shares, keys, method) -> np.ndarray        # original
```
Each recovery function also returns/records its runtime. `rect=None` means no MSA.

CLI demo:
```
python -m pss.cli demo --image data/images/sailboat.png --rect 300 300 60 40 --seed 123 --method edge_directed --out out/
```
Saves: original, masked, LR, shares 1–4 (expanded and final), and each recovered image; prints PASS/FAIL for exact equality at each stage and the bpp values.

### 6.1 Streamlit demo UI (`app.py`)

Built **only after milestone M3 passes** (pipeline exact on all stages). The UI calls the public API of §6 and contains no algorithm logic. Keep it to one file, about 150 lines.

Sidebar: master seed, predictor choice (`bilinear` / `edge_directed`), (r, n) fixed at (3, 4).

Tabs:
1. **Owner:** upload a grayscale image (auto-converted to 512×512), choose the MSA rectangle with four number inputs (top, left, height, width) and show a preview with the rectangle drawn. Button "Create shares". Shows the masked image, the 4 final shares (noise images), `bpp_paper`, `bpp_free` and timings.
2. **Data hider:** pick a share, type a text payload and a hiding key, button "Embed", then download the embedded share as PNG. A second button extracts and shows the text (with CRC status).
3. **Receiver:** multiselect which shares to use (1–4) and text boxes for the keys. Shows the result according to what was supplied: fewer than 3 shares gives an error message, 3 shares gives the LR image (Stage 1), 4 shares gives the masked image (Stage 2), 4 shares and the MSA key gives the original (Stage 3). Displays "exact match with original: yes/no" when the original is available.

4. **Metrics** (Tier 2): entropy, correlation, NPCR and UACI of the shares, and the histogram of the original vs a share.

Run with `streamlit run app.py`. Keep results in `st.session_state`. A tiny test (`tests/test_app_smoke.py`) imports the app helpers and runs owner to receiver once.

### 6.2 Neural networks
The paper's RCAN and GICA are **not** used: training or running them in 4 days is not feasible, and exact recovery would require bit-identical network output on sender and receiver, which is fragile across machines and library versions.

**Tier 3 stretch (only if everything else is finished):** a tiny CNN predictor (for example 3 convolution layers, about 10 k parameters, trained on 64×64 crops for about 20 minutes on CPU), used **only** to compare residual entropy against the integer predictors in E8. It must not be used inside the lossless pipeline unless its output is made integer-exact with fixed-point arithmetic, which is out of scope. Present it as an exploratory result, not as a reproduction of RCAN. In the report, list a fully integer, deterministic neural predictor as future work.

---

## 7. Test specification (`pytest`)

All tests use fixed seeds. Use 64×64 crops for speed; at least one test on a full 512×512 image.

**GF(2⁸)**
- `mul(0x53, 0xCA) == 0x01` under 0x11B and the inverse table is consistent under 0x11D (parametrise the polynomial).
- Field axioms on 10 000 random triples.

**Paper example (Fig. 5), polynomial 0x11D**
- c=73, a₁=76, a₂=106, K=110 → shares at x=110,111,112,113 are **236, 202, 53, 19**.
- c=77, a₁=78, a₂=184, K=6 → shares at x=6,7,8,9 are **97, 151, 38, 208**.
- If this fails, check the polynomial first (see D1).

**Sharing**
- Round trip for (3,4): all 4 combinations of 3 shares recover (c, a₁) exactly, 10⁵ random pairs.
- Round trip for (4,4).
- Wrong key → mismatch for ≥ 99 % of pairs. Too few shares with the k-solver → mismatch.

**Predictor** — properties listed in §5.5.

**Huffman**
- Round trip on random Laplace-distributed residuals, a constant image (single symbol), and a residual using all 511 symbols.
- Compressed length < raw 9-bit length for natural images.

**Embedding**
- Expansion → extraction of first pixels returns the shared image.
- Header parse round trip; capacity error is raised for an oversized MSA.
- Data hider: random payloads per share round trip; CRC detects a wrong key; after hiding, Stage 1–3 outputs are unchanged.

**Pipeline (the important ones)**
- For every image (64×64 crops and the 512×512 set), a random rectangle, and **every combination of 3 shares**: Stage 1 L equals original L.
- All 4 shares: Stage 2 masked equals owner's masked; Stage 3 equals original (`np.array_equal`).
- Both predictors; with and without MSA; MSA at image corners and covering a large area.

**Negative tests**
- 2 shares (r−1) treated with the k=2 solver → output ≠ L (record PSNR).
- 3 shares for Stage 2 → raises an error.
- Wrong `K_L`, wrong `K_C`, wrong `K_sc` → incorrect output or `InvalidKeyError`; no crash with silent success.
- Stage 3 without `K_sc` → only masked image available.

---

## 8. Experiments (`experiments/run_all.py`)

Tier 1 = E1–E6 (must). Tier 2 = E7–E10 (after Tier 1 passes). Master seed fixed and logged. Each experiment writes `results/E?.csv` and figures to `figures/`.

| ID | Experiment | Output |
|---|---|---|
| **E1** | Correctness matrix: 8 images × predictors × all 3-share combinations × 3 stages; also 20 random MSA rectangles per image | table of PASS/FAIL (must be all PASS) |
| **E2** | Embedding rate: per image, for both predictors (`bilinear`, `edge_directed`), without MSA; paper formula and practical formula | table + bar chart; add a column with the **paper's quoted** Table III value (labelled "quoted") |
| **E3** | Security metrics for expanded shares and final shares (4 shares each) and for the original: entropy (Eq. 18), horizontal/vertical adjacent correlation (Eq. 19), NPCR and UACI vs original (Eqs. 20–21) | table in the style of paper Tables I & II; ideal NPCR 99.6094 %, UACI 33.4635 % |
| **E4** | Histograms of original, masked, 4 final shares; visual figure like paper Fig. 8 (original, masked, LR, 4 shares, recovered images) | figure |
| **E5** | Negative/attack tests from §7: r−1 shares, wrong keys, missing keys | table, include PSNR/entropy of failed outputs |
| **E6** | Runtime per stage (owner: inpaint, predict, compress, share, embed; receiver: stages 1–3), mean ± std over 5 runs, 512×512 | table |
| **E7** (Tier 2) | MSA size sweep: rectangle sizes from 16×16 up to the largest that fits capacity; plot bpp_free vs MSA area; report the maximum MSA size | table + line plot |
| **E8** (Tier 2) | Predictor ablation: residual PDF (like paper Fig. 9, averaged over the 8 images) and embedding rate per image (like Fig. 10) for `bilinear`, `edge_directed`, `cv2_bicubic`; plus paper-faithful vs canonical Huffman header size | two figures + small table |
| **E9** (Tier 2) | Ramp-leakage check (D3): given r−1 shares *and* the true K, count how many (c, a₁) candidates remain for sample pairs (expected 256 of 65 536); also show that without K the shares look uniform | short table |
| **E10** (Tier 2) | Data hider: random payload up to capacity on every share, extract, verify equality and CRC; verify recovery unchanged; report capacity in bytes and bpp | table |

**Quote table** (labelled "from He et al. 2026, Table III, not re-run"), average embedding rate in bpp: Wu et al. 1.8403; Chen et al. [30] 0.5000; Chen et al. [31] 3.5000; Hua et al. [32] 2.3372; [32] full size 4.3072; Yu et al. 2.7383; Chen & Xu 3.3319; Qin et al. 1.0807; Hua et al. [36] 1.7401; PSS-DRDHEI (paper) 5.2288.

Expectations (not promises): smoother images (Airplane, Man) give higher bpp than noisy ones (Baboon, Boat); `edge_directed` should be ≥ `bilinear` on most images. **Report whatever is measured.**

---

## 9. Metric definitions

- **PSNR** = 10·log₁₀(255² / MSE); exact recovery → report "∞ (exact)".
- **SSIM**: `skimage.metrics.structural_similarity`.
- **Entropy** H = −Σ p_i log₂ p_i over 256 levels (ideal 8).
- **Adjacent correlation**: sample 10 000 random horizontally (or vertically) adjacent pixel pairs, Pearson correlation; report absolute value.
- **NPCR** = (1/MN) Σ D(m,n) × 100 %, D = 1 if pixels differ. **UACI** = (1/MN) Σ |I − I*| / 255 × 100 %.
- **Embedding rate** = bits embedded per original pixel, denominator M·N, as in §5.8.
- **Storage expansion**: total size of n final shares ÷ original size = n×. State it explicitly, as the scheme trades storage for redundancy.

---

## 10. Report outline and honesty rules

**Report sections:** 1 Introduction and problem; 2 Background (Shamir, GF(2⁸), RDHEI, prediction coding); 3 Scheme (flow diagram of paper Fig. 3, redrawn in your own words); 4 Implementation (what is simplified and why, table from §2); 5 Experiments and results (E1–E10); 6 Analysis, including discrepancies D1–D7; 7 Limitations and future work (replace predictor by a trained network, ensure determinism); 8 Conclusion; References.

**Rules**
- Call it a **simplified reproduction**. State clearly which numbers are yours and which are quoted from the paper.
- Do not claim improvements. The honest contributions are: working open implementation, verified lossless 3-stage recovery, the polynomial discrepancy (D1), key-range fix (D2), ramp-leakage observation (D3), a three-predictor ablation, a measured MSA-size sweep, and a measured check of the ramp-leakage observation.
- Cite the paper and every library. Write all text in your own words; do not copy paper sentences or figures. Redraw diagrams.
- Include `DEVIATIONS.md` as an appendix.

---

## 11. Plan: 4 days, 2 people, with milestones

| Day | Person A — crypto and embedding | Person B — image pipeline and experiments |
|---|---|---|
| **1** | **M1:** `gf256.py`, `sharing.py`, `keys.py`; pass Fig. 5 vectors and sharing round trips. | **M2:** `msa.py`, `predictor.py`, `huffman.py`; pass predictor and Huffman tests. |
| **2** | **M3:** `embed.py` (expand, layout, hider). Then pair with B on `pipeline.py`. | **M3:** `pipeline.py` end to end with A. **Milestone gate (end of day): one image recovers exactly at Stage 1, 2, 3.** |
| **3** | Negative tests, E5, security metrics E3/E4, then Tier 2: E9, E10. | `run_all.py`: E1, E2, E6, then Tier 2: E7, E8. **Streamlit UI (§6.1)** starts here: Owner and Receiver tabs. |
| **4** | Finish UI (Data hider tab, Metrics tab), UI smoke test, README, `DEVIATIONS.md`, slides (scheme + security). | Final report text, tables, slides (results), demo rehearsal, run everything once more from a clean checkout. |

Day 4 is the buffer. If the Day 2 gate slips, drop in this order: Tier 3, E9/E10, E8's bicubic and canonical parts, E7, the UI Metrics tab. Never drop Tier 1.

**Milestones with pass condition:** M1 = tests in `test_gf256.py`, `test_sharing.py` green. M2 = `test_predictor.py`, `test_huffman.py` green. M3 = `test_embed.py`, `test_pipeline.py`, `test_negative.py` green. M4 = `experiments/run_all.py` runs from scratch and regenerates every table and figure. M5 = `streamlit run app.py` demonstrates owner, hider and receiver flows end to end.

---

## 12. Acceptance checklist (definition of done)

- [ ] Fig. 5 numeric example reproduced exactly (shares 236/202/53/19 and 97/151/38/208).
- [ ] For all 8 images and all 4 three-share combinations, Stage 1 recovers L exactly.
- [ ] With all 4 shares, Stage 2 recovers the masked image exactly; with keys, Stage 3 recovers the original exactly (`np.array_equal`).
- [ ] Works with and without MSA; MSA touching borders; capacity error raised properly.
- [ ] Data hider embed and extract works per share and does not change recovery.
- [ ] Wrong or missing keys/shares fail as specified.
- [ ] Entropy ≈ 8, correlation ≈ 0, NPCR ≈ 99.6 %, UACI ≈ 33.4 % for shares (report actual numbers).
- [ ] Embedding rates reported for all images with both formulas, next to the paper's quoted values.
- [ ] `streamlit run app.py` demonstrates Owner, Data hider and Receiver tabs (all three stages).
- [ ] Tier 2 done where time allowed: E7–E10 results in `results/`, each with a one-paragraph interpretation in the report.
- [ ] One command reproduces all tables and figures: `python experiments/run_all.py`.
- [ ] `README.md` with install, run, test instructions; `DEVIATIONS.md` filled in.
- [ ] Both team members can explain: GF(2⁸) sharing, why 2 pixels per polynomial, how the residual is compressed, where the free room comes from, and what each stage needs.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| GF arithmetic bug breaks everything silently | Fig. 5 vectors and field-axiom tests on Day 1 |
| Predictor not bit-exact between encode and decode | Integer-only predictors; same function used by owner and receiver; determinism test |
| Huffman decode too slow in Python | Accept up to ~10 s per 512² image; optional bit-trick decoding if time permits |
| Embedding rate lower than the paper's | Expected (simpler predictor); report honestly; never tune the report to match |
| Time overrun | Gate at end of Day 2; drop optional experiments first |
| Image licences | Document sources in `data/README.md`; do not redistribute if forbidden |

---

## 14. References

[1] M. He, X. Xie, X. Wang, L. Zhang, "Progressive Secret Sharing Through the Integration of Deep Learning and Reversible Data Hiding in Encrypted Images," *IEEE Internet Things J.*, vol. 13, no. 3, pp. 4564–4577, 2026. doi:10.1109/JIOT.2025.3635284.
[2] A. Shamir, "How to share a secret," *Commun. ACM*, vol. 22, no. 11, pp. 612–613, 1979.
[3] J. Yu et al., "Generative image inpainting with contextual attention," CVPR 2018 (GICA — replaced here).
[4] Y. Zhang et al., "Image super-resolution using very deep residual channel attention networks," ECCV 2018 (RCAN — replaced here).
[5] D. M. Thodi and J. J. Rodriguez, "Prediction-error based reversible watermarking," ICIP 2004 (MED predictor).
[6] P. Bas, T. Filler, T. Pevný, "Break our steganographic system: The ins and outs of organizing BOSS," Information Hiding 2011 (BOSSbase).
[7] OpenCV, NumPy, scikit-image documentation (library citations).
