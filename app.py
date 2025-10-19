import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Sales Prediction App", page_icon="📊")
st.title("📊 Sales Prediction App")
st.write("ใส่งบโฆษณาแต่ละช่องทางเพื่อประมาณการยอดขาย (sales)")

MODEL_FILE = Path("model-reg-xxx.pkl")  # ใช้ชื่อไฟล์ .pkl ของคุณ
with MODEL_FILE.open("rb") as f:
    model = pickle.load(f)

youtube = st.number_input("YouTube budget", min_value=0.0, value=50.0, step=1.0)
tiktok = st.number_input("TikTok budget", min_value=0.0, value=50.0, step=1.0)
instagram = st.number_input("Instagram budget", min_value=0.0, value=50.0, step=1.0)

if st.button("Predict"):
    df = pd.DataFrame({"youtube":[youtube], "tiktok":[tiktok], "instagram":[instagram]})
    yhat = model.predict(df)
    st.success(f"📈 Estimated Sales: {float(yhat[0]):.2f}")
