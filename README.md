# Progressive Secret Image Sharing

This repository is a **simplified reproduction** of the PSS-DRDHEI scheme described in `SPEC.md`.
It uses deterministic integer predictors, OpenCV Telea inpainting, GF(2^8) pair sharing, residual
Huffman coding, redundant-pixel embedding, and a four-share `(3, 4)` progressive recovery flow.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
```

## Tests

```bash
python -m pytest -q
```

For the complete manual and automated test procedure, see `TESTING.md`.

## CLI demo

```bash
python -m pss.cli demo --image data/images/sailboat.png \
  --rect 300 300 60 40 --seed 123 --method edge_directed --out out
```

## Streamlit demo

```bash
streamlit run app.py
```

## React research dashboard

The polished local dashboard lives in `web/` and uses the tested Python pipeline through FastAPI.
Run the API from the repository root:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The dashboard provides an interactive MSA rectangle selector, live
capacity feedback, predictor selection, share integrity status, progressive receiver stages, data
hider controls, and security metrics. The Streamlit app remains available as a lightweight fallback.

## Experiments

```bash
python experiments/run_all.py
```

The repository includes eight local public sample substitutes under `data/images/`, using the
filenames expected by the experiment runner. Their provenance is recorded in `data/README.md` and
the run source is recorded in `results/run_metadata.json`. They are not the exact paper images.

The project also includes a MED predictor comparison, keyed integrity checks for detecting
modified shares, adaptive MSA placement, and resilient three-share recovery for the compressed-code
and MSA streams. These are project extensions and are not claimed as an exact reproduction of the
paper.

## Research extensions

The Owner workflow supports three MSA modes: manual selection, automatic texture selection, and
automatic edge selection. Adaptive modes keep the requested dimensions and choose a reproducible
content-aware window. The Receiver workflow can automatically verify all available shares, exclude
invalid or missing shares, and recover from the first valid three-share threshold subset.

The experiment runner now writes:

- `results/E12.csv` — manual versus adaptive MSA placement and exact-recovery results.
- `results/E13.csv` — one-tampered-share and one-missing-share resilient recovery results across
  progressive stages.
