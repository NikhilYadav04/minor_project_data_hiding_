# Testing guide

This guide tests the completed **simplified reproduction and project extensions**. It does not test
GICA or RCAN because those paper components are intentionally replaced by OpenCV Telea inpainting
and deterministic integer predictors. The tests verify the behavior that is actually implemented.

## Current verified status

The latest local verification is:

```text
Python test suite: 30 passed
React/Vite production build: successful
Experiment outputs: E1–E16 generated
```

## 1. Install the project

From the repository root:

```bash
python -m pip install -e ".[test]"
```

Check the runtime:

```bash
python --version
python -c "import numpy, cv2, pytest; print(numpy.__version__, cv2.__version__, pytest.__version__)"
```

## 2. Run the automated tests

Run the complete suite:

```bash
python -m pytest -q
```

The suite covers:

- GF(2^8) arithmetic and the paper's Fig. 5 vectors.
- All four 3-of-4 share combinations.
- Predictor properties and residual reconstruction.
- Bilinear, edge-directed, MED, bicubic, and automatic predictor comparisons.
- Huffman round trips, including constant and all-symbol inputs.
- Expansion, core payload extraction, data-hider CRC validation.
- Stage 1, Stage 2, and Stage 3 recovery.
- MSA and no-MSA cases.
- Wrong-key, insufficient-share, and capacity-error behavior.
- Keyed integrity verification and tampered-share detection.
- Presence, dimensions, and dtype of the eight bundled test images.
- Owner-to-receiver backend smoke behavior used by the UI.
- Adaptive MSA selection, automatic predictor selection, resilient recovery, and extension APIs.

## 3. Run a deterministic CLI demonstration

Create a temporary 512x512 image:

```bash
python -c "import cv2,numpy as np; y,x=np.indices((512,512)); cv2.imwrite('demo_input.png',((7*x+13*y+x*y%31)%256).astype('uint8'))"
```

Run the complete owner and receiver flow:

```bash
python -m pss.cli demo \
  --image demo_input.png \
  --rect 300 300 60 40 \
  --seed 123 \
  --method edge_directed \
  --out demo_out
```

Expected output includes:

```text
Stage 2 exact: True
Stage 3 exact: True
```

The output directory should contain the original, masked, LR, expanded-share, final-share, and
recovered images. Remove the temporary files afterward if desired:

```bash
rm -rf demo_input.png demo_out
```

On Windows PowerShell, use `Remove-Item -Recurse -Force demo_out`.

## 4. Verify the eight bundled images

The project includes these local files:

```text
data/images/airplane.png
data/images/baboon.png
data/images/boat.png
data/images/couple.png
data/images/elaine.png
data/images/man.png
data/images/peppers.png
data/images/sailboat.png
```

They are 512x512 grayscale public sample substitutes, not the exact paper images. Their source
mapping is documented in `data/README.md`.

Verify them with:

```bash
python -m pytest tests/test_data_images.py -q
```

Expected result:

```text
1 passed
```

## 5. Run the experiment suite

```bash
python experiments/run_all.py
```

Expected files:

```text
results/E1.csv
results/E2.csv
results/E3.csv
results/E5.csv
results/E6.csv
results/E7.csv
results/E8.csv
results/E9.csv
results/E10.csv
results/E11.csv
results/E12.csv
results/E13.csv
results/E14.csv
results/E15.csv
results/E16.csv
results/run_metadata.json
figures/E2_bpp.png
figures/E4_overview.png
figures/E7_msa_sweep.png
figures/E8_residuals.png
figures/E8_embedding_rate.png
```

`E8.csv` contains the predictor comparison, including MED.

`E11.csv` contains the tampered-share detection results. E12-E16 contain the project-extension
experiments described in Sections 12 and 14.

Check the correctness and data-hider results:

```bash
python -c "import pandas as pd; e1=pd.read_csv('results/E1.csv'); e10=pd.read_csv('results/E10.csv'); e11=pd.read_csv('results/E11.csv'); print('E1:', e1[['stage1_pass','stage2_pass','stage3_pass']].all().all()); print('E10:', e10['round_trip'].all()); print('E11:', e11['tampered_share_rejected'].all())"
```

All three values should be `True`.

Confirm the image source and dimensions in:

```text
results/run_metadata.json
```

The expected source value for the bundled files is:

```json
"bundled_public_substitute"
```

## 6. Test MED and predictor selection

Run the predictor-specific tests:

```bash
python -m pytest tests/test_predictor.py -q
```

Run a MED CLI demonstration:

```bash
python -m pss.cli demo \
  --image data/images/sailboat.png \
  --rect 300 300 60 40 \
  --seed 123 \
  --method med \
  --out med_demo_out
```

Expected output:

```text
Stage 2 exact: True
Stage 3 exact: True
```

The MED predictor is a deterministic integer predictor. It is an original lightweight extension for
this project and is not claimed to be the RCAN network used in the paper.

## 7. Test tampered-share detection

Run the tamper tests:

```bash
python -m pytest tests/test_tamper.py -q
```

The tests verify that:

- Four untouched shares are valid.
- Modifying one share causes integrity verification to fail.
- The modified share is excluded by `find_valid_shares`.
- Recovery does not silently accept the corrupted four-share set.
- Data-hider embedding in the free area does not invalidate the share tag.

The experiment result is stored in:

```text
results/E11.csv
```

Each row should report:

```text
clean_valid_count = 4
tampered_valid_count = 3
tampered_share_rejected = True
```

## 8. Test the Streamlit UI

Start the app:

```bash
streamlit run app.py
```

In the browser:

1. Upload a grayscale PNG or JPEG.
2. Leave the default predictor as `edge_directed`.
3. Set an MSA rectangle, for example top `300`, left `300`, height `60`, width `40`.
4. Click **Create shares**.
5. Confirm that the masked image, four final shares, and embedding rates appear.
6. In **Data hider**, choose a share, enter text such as `test payload`, and click **Embed payload**.
7. Click **Extract payload** and confirm the same text is returned.
8. In **Receiver**, select three shares and confirm an LR image appears.
9. Select all four shares and confirm the masked and original images appear.
10. In **Metrics**, confirm entropy, correlation, NPCR, and UACI rows are displayed.
11. Repeat share creation with predictor `med` and confirm Stage 2 and Stage 3 remain exact.
12. Modify a generated share file or use the tamper test and confirm the receiver reports the share
    as tampered rather than treating it as valid.

The backend UI smoke test is included in `tests/test_app_smoke.py`; the manual steps above verify the
interactive browser behavior.

## 9. Test the React research dashboard

Start the FastAPI backend from the repository root:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

In a second terminal, start the frontend:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Verify this flow:

1. Open **Owner**.
2. Upload one of the eight files in `data/images/`.
3. Drag across the image to create the amber MSA rectangle.
4. Confirm the rectangle dimensions update while dragging.
5. Choose `Bilinear`, `Edge directed`, `MED`, and `Automatic best predictor` in separate runs.
6. Try Manual, Automatic · texture, and Automatic · edges MSA modes. Confirm the selected
   coordinates are displayed after **Create protected shares**.
7. Confirm the masked image and four shares appear after **Create protected shares**.
8. Open **Data hider**, add a short text payload, and extract it again.
9. Open **Receiver** and run Stage 1, Stage 2, and Stage 3. The dashboard uses automatic
   valid-share selection for all stages and reports which threshold subset was used.
10. Confirm each result reports an exact match and shows the used valid-share indexes.
11. Open **Analysis** and confirm the entropy chart and share metrics table appear.
12. Open **Research Summary** and confirm the baseline, extensions, predictor decision, MSA
    coordinates, integrity state, and conclusion are shown.

The backend integration test can be run without starting the server manually:

```bash
python -m pytest tests/test_api.py -q
```

## 10. Boundary and failure checks

The following should fail clearly rather than silently succeeding:

- The strict legacy Stage 2 endpoint with only two or three shares. The research-extension
  `/api/receiver/recover-auto` endpoint intentionally accepts any three valid shares.
- Recovery with a wrong master seed.
- Data-hider extraction with a wrong hiding key.
- An MSA whose payload exceeds the redundant capacity.
- An image with odd height or a width not divisible by four.
- A modified share with an invalid integrity tag.

The automated suite covers these cases. To inspect a specific test:

```bash
python -m pytest tests/test_negative.py -q
```

## 11. Test interpretation

The strongest correctness checks are exact array comparisons, not visual similarity:

```python
np.array_equal(stage2, owner_output.masked)
np.array_equal(stage3, original)
```

For exact recovery, PSNR should be reported as `∞ (exact)`. Security metrics such as entropy,
correlation, NPCR, and UACI are measurements of the generated shares and should not be presented as
proof of the security of the original paper's neural-network implementation.


## 12. Test the research extensions

Run the extension tests:

```bash
python -m pytest tests/test_research_extensions.py -q
```

They verify that adaptive texture and edge selectors are reproducible, the selected MSA remains
inside the image, and masked/original recovery succeeds with one tampered or missing share. Two
tampered shares are rejected because fewer than three valid shares remain.

Run the API extension test:

```bash
python -m pytest tests/test_api.py -q
```

The API test verifies `msa_mode=auto_texture` and `/api/receiver/recover-auto`.

After running the experiment suite, inspect:

```text
results/E12.csv
results/E13.csv
```

E12 compares manual, texture-based, and edge-based MSA placement. E13 reports automatic recovery
from one tampered share and one missing share for Stages 1, 2, and 3.


## 13. Test automatic predictor selection

Run:

```bash
python -m pytest tests/test_research_extensions.py -q
python -m pytest tests/test_api.py -q
```

Choose `Automatic best predictor` in the React Owner workflow. The run should report the selected
predictor and a comparison table containing bilinear, edge-directed, and MED compressed bit
lengths. The selected method is the candidate with the shortest compressed residual stream, and
Stage 3 must remain exact.

## 14. Test the complete research experiment matrix

Run the experiment module directly if the full suite is already familiar:

```bash
python -c "import experiments.run_all as r; images=r.load_images(); r.e14_auto_predictor(images); r.e15_threshold_matrix(images); r.e16_attack_matrix(images)"
```

Inspect:

```text
results/E14.csv
results/E15.csv
results/E16.csv
```

E14 records the automatic predictor decision and exact recovery. E15 tests all four combinations of
three available shares for Stages 1, 2, and 3. E16 covers one-pixel tampering, block tampering, one
missing share, two missing shares, wrong master key, wrong data-hider key, and a corrupted data-hider
payload. The expected interpretation is recovery with one invalid/unavailable share and rejection
when fewer than three valid shares remain.

## 15. Use the Research Summary view

Open the fifth dashboard step after creating shares. It summarizes the baseline reproduction, the
three project extensions, the selected predictor, MSA coordinates, embedding room, integrity state,
and the exact-recovery conclusion for the current live run.


## 16. Documentation consistency check

Before a presentation, confirm these files agree with the implementation:

- `README.md` describes the current React/FastAPI dashboard and E1–E16 outputs.
- `SPEC.md` distinguishes the paper-inspired baseline from project extensions.
- `DEVIATIONS.md` records Telea, integer predictors, adaptive MSA, automatic predictor selection,
  and three-share resilient code/MSA recovery.
- `data/README.md` identifies the bundled images as public substitutes.

The README is the professor-facing entry point; `TESTING.md` is the reproducibility procedure;
`SPEC.md` and `DEVIATIONS.md` provide the technical and academic context.
