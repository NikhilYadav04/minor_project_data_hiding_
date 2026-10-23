# Image data

Place the eight 512x512 grayscale PNG files named in `SPEC.md` in `data/images/`:

`airplane.png`, `baboon.png`, `boat.png`, `couple.png`, `elaine.png`, `man.png`, `peppers.png`,
`sailboat.png`.

The repository currently contains eight local 512x512 grayscale **public sample substitutes** with
the required filenames. They were generated from the scikit-image sample-image distribution and
are not the exact images shown in the paper. The mapping is documented below:

| Project filename | Bundled source sample |
|---|---|
| `airplane.png` | `skimage.data.camera` |
| `baboon.png` | `skimage.data.chelsea` converted to grayscale |
| `boat.png` | `skimage.data.rocket` converted to grayscale |
| `couple.png` | `skimage.data.astronaut` converted to grayscale |
| `elaine.png` | `skimage.data.coins` |
| `man.png` | `skimage.data.horse` |
| `peppers.png` | `skimage.data.coffee` converted to grayscale |
| `sailboat.png` | `skimage.data.moon` |

These files are suitable for implementation testing and reproducible demonstrations. They must be
labelled as substitutes in the report. If the exact paper images are later obtained with permission,
replace these files and update this table.

The files must be 8-bit grayscale PNGs. The pipeline will resize them to 512x512 when loaded by the
CLI or Streamlit UI. The experiment runner records whether each image came from `data/images/` or
from its deterministic synthetic fallback in `results/run_metadata.json`.
