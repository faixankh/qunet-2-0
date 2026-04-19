from __future__ import annotations
import streamlit as st
import torch
from PIL import Image
import numpy as np
from ..models.qunet2 import QUNet2

st.set_page_config(page_title="QUNet 2.0", layout="wide")
st.title("QUNet 2.0")
st.caption("Multimodal retinal imaging demo with segmentation, grading, and uncertainty.")

model = QUNet2()
model.eval()

uploaded = st.file_uploader("Upload a retinal image", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB").resize((224, 224))
    arr = np.asarray(image).astype("float32") / 255.0
    image_t = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    oct_t = torch.zeros(1, 1, 112, 112)
    with torch.no_grad():
        out = model({"image": image_t, "oct": oct_t})
    st.image(image, caption="Input", use_container_width=True)
    st.write("Classification logits:", out["cls_logits"].squeeze(0).tolist())
    st.write("Uncertainty logits:", out["log_var_logits"].squeeze(0).tolist())
