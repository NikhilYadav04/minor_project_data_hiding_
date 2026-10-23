import numpy as np
import cv2
from pathlib import Path
from streamlit.testing.v1 import AppTest

from app import prepare_upload
from pss.keys import KeyBundle
from pss.pipeline import owner_encode, recover_stage3


def test_owner_to_receiver_backend_smoke():
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    keys = KeyBundle(17)
    output = owner_encode(image, (4, 4, 8, 8), keys)
    assert np.array_equal(recover_stage3(output.final_shares, keys), image)


def test_streamlit_initial_screen_smoke():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert [title.value for title in app.title] == ["Progressive Secret Image Sharing"]
    assert len(app.tabs) == 4


def test_streamlit_owner_flow_smoke():
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py")
    app.session_state["image"] = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    app.run(timeout=30)
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert [success.value for success in app.success] == ["Shares created"]
    assert "output" in app.session_state
    assert len(app.session_state["output"].final_shares) == 4
    assert len(app.dataframe) == 2
