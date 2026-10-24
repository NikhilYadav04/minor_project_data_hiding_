# Deviations and implementation decisions

This project follows `SPEC.md` and labels measured values separately from values quoted from the
paper. The following decisions are intentional:

1. The default irreducible polynomial is `0x11D`, because it reproduces the paper's worked example;
   `0x11B` remains available for parametrized GF tests.
2. Evaluation keys are sampled from `1..256-n`, avoiding zero and out-of-range points.
3. GICA is replaced by OpenCV Telea inpainting and RCAN is replaced by integer predictors.
4. The Huffman matching-rule width includes zero with `ceil(log2(max_len + 1))`.
5. Canonical Huffman mode uses the high bit of the one-byte width marker as an internal format flag;
   the default paper-faithful format leaves that bit clear.
6. The supplied paper discrepancies D1-D7 are observations for the report, not silently corrected
   claims about the original publication.

7. The project extension adds `manual`, `auto_texture`, and `auto_edge` MSA placement.
   Adaptive modes keep the requested rectangle dimensions and select the highest-scoring fixed-size
   window using local variance or Sobel gradient magnitude; they are an experimental heuristic, not
   a claim about the paper's original MSA selector.
8. The project extension changes the compressed-code and MSA payload sharing from 4-of-4 to 3-of-4
   threshold sharing. This preserves the `(3, 4)` recovery contract and enables masked/original
   recovery after one missing or tampered share. The strict legacy recovery functions still require
   four supplied shares, while `recover_available` selects and reports a valid threshold subset.
9. E12 and E13 are project extension experiments for adaptive MSA and resilient recovery; their
   results must be reported separately from paper-quoted values.

10. The project extension adds automatic predictor selection. It evaluates bilinear,
    edge-directed, and MED on the masked image and chooses the candidate with the shortest
    paper-faithful compressed residual stream, using mean absolute residual as a deterministic
    tie-breaker. This is an adaptive project heuristic, not the RCAN model from the paper.
11. E14, E15, and E16 extend the evaluation: automatic predictor selection, all four threshold
    combinations, and a recovery/data-hider attack matrix. They should be reported as extension
    evidence rather than as measurements from the original publication.
