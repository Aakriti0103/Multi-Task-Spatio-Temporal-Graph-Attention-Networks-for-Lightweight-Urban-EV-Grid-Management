import streamlit as st
import numpy as np
import pandas as pd
import torch

from models.grid_management import MultiTaskSTGAT

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = MultiTaskSTGAT()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("⚡ EV Grid Prediction System")

# -----------------------------
# LOAD DATA (USE YOUR REAL DATA)
# -----------------------------
@st.cache_data
def load_sample():
    df_occ = pd.read_csv("data/occupancy.csv")
    df_dur = pd.read_csv("data/duration.csv")
    df_vol = pd.read_csv("data/volume.csv")

    # drop time column
    df_occ = df_occ.drop(columns=[df_occ.columns[0]])
    df_dur = df_dur.drop(columns=[df_dur.columns[0]])
    df_vol = df_vol.drop(columns=[df_vol.columns[0]])

    occ = df_occ.values
    dur = df_dur.values
    vol = df_vol.values

    # take last 24 hours
    occ = occ[-24:]
    dur = dur[-24:]
    vol = vol[-24:]

    data = np.stack([occ, dur, vol], axis=-1)  # (24, 275, 3)
    return data

# -----------------------------
# BUTTON
# -----------------------------
if st.button("Run Prediction on Latest Data"):

    sample = load_sample()

    st.write("Input shape:", sample.shape)

    # Convert to tensor
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

    # ⚠️ IMPORTANT: No mask for now
    pred_occ, pred_dur, pred_vol, _ = model(x, None)

    pred_occ = pred_occ.detach().numpy()[0]
    pred_dur = pred_dur.detach().numpy()[0]
    pred_vol = pred_vol.detach().numpy()[0]

    st.subheader("📊 Predictions (Next Hour)")

    st.write("Occupancy (first 10 stations):")
    st.write(pred_occ[:10])

    st.write("Duration (first 10 stations):")
    st.write(pred_dur[:10])

    st.write("Volume (first 10 stations):")
    st.write(pred_vol[:10])