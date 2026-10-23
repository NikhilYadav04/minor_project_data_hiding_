from __future__ import annotations

import cv2
import numpy as np

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows importing helpers without Streamlit installed
    st = None

from pss.embed import hider_embed, hider_extract
from pss.auth import verify_share
from pss.keys import KeyBundle
from pss.metrics import adjacent_correlation, entropy, npcr, uaci
from pss.pipeline import owner_encode, recover_stage1, recover_stage2, recover_stage3


def prepare_upload(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("uploaded file is not a readable image")
    return cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)


def main() -> None:  # pragma: no cover - interactive UI
    if st is None:
        raise RuntimeError("Install Streamlit to run the UI")
    st.set_page_config(page_title="PSS image sharing", layout="wide")
    st.title("Progressive Secret Image Sharing")
    seed = st.sidebar.number_input("Master seed", min_value=0, value=123, step=1)
    method = st.sidebar.selectbox("Predictor", ["edge_directed", "bilinear", "med"])
    uploaded = st.sidebar.file_uploader("Grayscale image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            image = prepare_upload(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["image"] = image
    image = st.session_state.get("image")
    tabs = st.tabs(["Owner", "Data hider", "Receiver", "Metrics"])
    with tabs[0]:
        if image is None:
            st.info("Upload an image to begin")
        else:
            st.image(image, caption="Original", clamp=True)
            top = st.number_input("MSA top", 0, 511, 300)
            left = st.number_input("MSA left", 0, 511, 300)
            height = st.number_input("MSA height", 1, 512 - int(top), 60)
            width = st.number_input("MSA width", 1, 512 - int(left), 40)
            if st.button("Create shares"):
                keys = KeyBundle(int(seed))
                output = owner_encode(image, (int(top), int(left), int(height), int(width)), keys, method)
                st.session_state.update({"keys": keys, "output": output})
                st.success("Shares created")
            output = st.session_state.get("output")
            if output:
                st.image(output.masked, caption="Masked image", clamp=True)
                st.write({"bpp_paper": output.stats["bpp_paper"], "bpp_free": output.stats["bpp_free"]})
                st.image(output.final_shares, caption=[f"Final share {i}" for i in range(1, 5)], clamp=True)
    with tabs[1]:
        output = st.session_state.get("output")
        keys = st.session_state.get("keys")
        if output and keys:
            share_index = st.selectbox("Share", range(4))
            text = st.text_area("Payload")
            dh_key = st.number_input("Hiding key", min_value=0, value=keys.dh_seed(share_index), step=1)
            if st.button("Embed payload"):
                output.final_shares[share_index] = hider_embed(output.final_shares[share_index], text.encode(), int(dh_key))
                st.success("Payload embedded")
            if st.button("Extract payload"):
                try:
                    st.code(hider_extract(output.final_shares[share_index], int(dh_key)).decode())
                except ValueError as exc:
                    st.error(str(exc))
        else:
            st.info("Create shares in the Owner tab first")
    with tabs[2]:
        output = st.session_state.get("output")
        keys = st.session_state.get("keys")
        if output and keys:
            selected = st.multiselect("Shares", [1, 2, 3, 4], default=[1, 2, 3])
            if len(selected) >= 3:
                chosen = [output.final_shares[i - 1] for i in selected]
                status_rows = [{"share": i, "integrity": "valid" if verify_share(output.final_shares[i - 1], i - 1, keys.auth_seed(i - 1)) else "tampered"} for i in selected]
                st.dataframe(status_rows)
                if len(selected) == 3:
                    st.image(recover_stage1(chosen, [i - 1 for i in selected], keys, method), caption="Stage 1 LR")
                else:
                    st.image(recover_stage2(output.final_shares, keys, method), caption="Stage 2 masked")
                    st.image(recover_stage3(output.final_shares, keys, method), caption="Stage 3 original")
        else:
            st.info("Create shares in the Owner tab first")
    with tabs[3]:
        output = st.session_state.get("output")
        if output:
            rows = []
            for index, share in enumerate(output.final_shares, 1):
                rows.append({"share": index, "entropy": entropy(share), "h_corr": adjacent_correlation(share, 1), "v_corr": adjacent_correlation(share, 0), "npcr_vs_original": npcr(image, share), "uaci_vs_original": uaci(image, share)})
            st.dataframe(rows)


if __name__ == "__main__":
    main()
