import numpy as np

from pss.deep_models import model_metadata
from pss.inpainting import contextual_inpaint, telea_inpaint
from pss.keys import KeyBundle
from pss.pipeline import owner_encode, recover_stage3


def test_contextual_inpainting_is_deterministic_and_preserves_context(image64):
    mask = np.zeros_like(image64, dtype=np.uint8)
    mask[16:32, 20:36] = 255
    first = contextual_inpaint(image64, mask)
    second = contextual_inpaint(image64, mask)
    assert np.array_equal(first, second)
    assert np.array_equal(first[mask == 0], image64[mask == 0])
    assert first.dtype == np.uint8


def test_telea_and_contextual_are_valid_backends(image64):
    mask = np.zeros_like(image64, dtype=np.uint8)
    mask[10:20, 10:20] = 255
    assert telea_inpaint(image64, mask).shape == image64.shape
    assert contextual_inpaint(image64, mask).shape == image64.shape


def test_paper_fidelity_pipeline_recovers_exactly(image64):
    output = owner_encode(image64, (8, 8, 12, 12), KeyBundle(811), "rcan_style", inpainting_method="contextual")
    recovered = recover_stage3(output.final_shares, KeyBundle(811), "rcan_style")
    assert np.array_equal(recovered, image64)
    assert output.stats["inpainting_method"] == "contextual"
    assert output.stats["predictor_backend"] == "rcan_style_fixed_point"


def test_rcan_style_metadata_is_stable():
    metadata = model_metadata()
    assert metadata["residual_groups"] == 5
    assert metadata["blocks_per_group"] == 5
    assert len(str(metadata["artifact_sha256"])) == 64
