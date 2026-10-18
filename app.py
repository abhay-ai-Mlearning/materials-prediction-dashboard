import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Materials Prediction Dashboard", page_icon="🧪", layout="wide")
BASE = Path(__file__).resolve().parent
st.title("🧪 Materials Prediction Dashboard")
st.caption("Browser-friendly and mobile-friendly materials predictions with interactive graphs.")

required = [
    "yield_strength_random_forest.joblib",
    "gamma_prime_radius_model.joblib",
    "gamma_prime_volume_fraction_model.joblib",
    "gamma_prime_spacing_model.joblib",
    "yield_strength_model_features.txt",
    "gamma_prime_model_metadata.json",
    "yield_strength_final_training_rows.csv",
    "gamma_prime_simulated_cleaned.csv",
]
missing = [f for f in required if not (BASE / f).exists()]
if missing:
    st.error("Missing files. Upload these beside app.py in your GitHub repository:")
    st.code("\\n".join(missing))
    st.info("Follow the README.md included with this project.")
    st.stop()

@st.cache_resource
def load_models():
    m = {
        "yield": joblib.load(BASE / "yield_strength_random_forest.joblib"),
        "radius": joblib.load(BASE / "gamma_prime_radius_model.joblib"),
        "volume": joblib.load(BASE / "gamma_prime_volume_fraction_model.joblib"),
        "spacing": joblib.load(BASE / "gamma_prime_spacing_model.joblib"),
    }
    yf = (BASE / "yield_strength_model_features.txt").read_text(encoding="utf-8").splitlines()
    meta = json.loads((BASE / "gamma_prime_model_metadata.json").read_text(encoding="utf-8"))
    gf = meta.get("feature_columns", [
        "aging_temperature_c", "log10_aging_time_hours",
        "nickel_content_wt_percent", "aluminum_content_wt_percent",
        "titanium_content_wt_percent", "tantalum_content_wt_percent"
    ])
    return m, yf, gf

@st.cache_data
def load_data():
    return (
        pd.read_csv(BASE / "yield_strength_final_training_rows.csv"),
        pd.read_csv(BASE / "gamma_prime_simulated_cleaned.csv")
    )

models, yield_features, gamma_features = load_models()
yd, gd = load_data()
yd.columns = [str(c).strip() for c in yd.columns]
gd.columns = [str(c).strip().lower() for c in gd.columns]
TC, YC = "temperature_(C)", "yield_strength_(MPa)"
composition_cols = [
    "Ni (wt%)", "Cr (wt%)", "Fe (wt%)", "Mo (wt%)", "Al (wt%)",
    "W (wt%)", "Ti (wt%)", "Nb (wt%)", "C (wt%)", "B (wt%)",
    "Mn (wt%)", "Si (wt%)", "Co (wt%)", "Ta (wt%)", "Re (wt%)"
]
if not {"Alloy_name", TC, YC}.issubset(yd.columns):
    st.error(f"Yield data must contain Alloy_name, {TC}, and {YC}. Found: {list(yd.columns)}")
    st.stop()
if "aging_temperature_c" not in gd.columns:
    st.error("The gamma-prime CSV is missing aging_temperature_c.")
    st.stop()

st.sidebar.header("Prediction inputs")
alloys = sorted(yd["Alloy_name"].dropna().astype(str).unique())
if not alloys:
    st.error("No alloy names found in the yield dataset.")
    st.stop()
default_index = alloys.index("Alloy_A") if "Alloy_A" in alloys else 0
alloy = st.sidebar.selectbox("Yield-strength alloy", alloys, index=default_index)
valid_t = pd.to_numeric(yd[TC], errors="coerce").dropna()
tmin = int(max(0, np.floor(valid_t.min()/10)*10))
tmax = int(min(1500, np.ceil(valid_t.max()/10)*10))
if tmax <= tmin: tmax = tmin + 100
yield_temp = st.sidebar.slider("Yield temperature (°C)", tmin, tmax, int(np.clip(700,tmin,tmax)), 10)

st.sidebar.subheader("Composition for γ′ model (wt%)")
ni = st.sidebar.number_input("Nickel", 0.0, 100.0, 75.0, 0.5)
al = st.sidebar.number_input("Aluminium", 0.0, 100.0, 5.5, 0.1)
ti = st.sidebar.number_input("Titanium", 0.0, 100.0, 2.0, 0.1)
ta = st.sidebar.number_input("Tantalum", 0.0, 100.0, 3.0, 0.1)

st.sidebar.subheader("γ′ aging conditions")
amin = int(max(0, np.floor(pd.to_numeric(gd["aging_temperature_c"], errors="coerce").min()/10)*10))
amax = int(np.ceil(pd.to_numeric(gd["aging_temperature_c"], errors="coerce").max()/10)*10)
if amax <= amin: amax = amin + 100
aging_temp = st.sidebar.slider("Aging temperature (°C)", amin, amax, int(np.clip(850,amin,amax)), 10)
aging_time = st.sidebar.number_input("Aging time (hours)", 0.1, 10000.0, 100.0, 1.0)

arows = yd[yd["Alloy_name"].astype(str) == alloy]
comp = {}
for col in composition_cols:
    vals = pd.to_numeric(arows[col], errors="coerce").dropna() if col in arows else pd.Series(dtype=float)
    comp[col] = float(vals.median()) if len(vals) else 0.0
yvals = {f: (float(yield_temp) if f == TC else comp.get(f, 0.0)) for f in yield_features}
yhat = float(models["yield"].predict(pd.DataFrame([yvals], columns=yield_features))[0])
gvals = {
    "aging_temperature_c": float(aging_temp),
    "log10_aging_time_hours": float(np.log10(aging_time)),
    "nickel_content_wt_percent": float(ni),
    "aluminum_content_wt_percent": float(al),
    "titanium_content_wt_percent": float(ti),
    "tantalum_content_wt_percent": float(ta),
}
gx = pd.DataFrame([{f: gvals.get(f, 0.0) for f in gamma_features}], columns=gamma_features)
rhat = float(models["radius"].predict(gx)[0])
vhat = float(models["volume"].predict(gx)[0])
shat = float(models["spacing"].predict(gx)[0])

st.subheader("Predicted values")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Yield strength", f"{yhat:,.2f} MPa")
c2.metric("γ′ particle radius", f"{rhat:,.2f} nm")
c3.metric("γ′ volume fraction", f"{vhat:,.2f}%")
c4.metric("Interparticle spacing", f"{shat:,.2f} nm")
st.caption(f"Yield prediction uses median recorded composition for {alloy}. γ′ estimates use the separate composition inputs; these are not a coupled simulation.")
st.warning("The γ′ models use synthetic illustrative training data and have weak test performance. These estimates are not experimentally validated and must not be used for engineering decisions.")

# Yield curve at the selected alloy's median recorded composition.
temps = np.linspace(float(valid_t.min()), float(valid_t.max()), 100)
yc = pd.DataFrame([{f: (float(t) if f == TC else comp.get(f,0.0)) for f in yield_features} for t in temps])
yc["predicted_mpa"] = models["yield"].predict(yc[yield_features])
ydplot = yd.copy()
ydplot[TC] = pd.to_numeric(ydplot[TC], errors="coerce")
ydplot[YC] = pd.to_numeric(ydplot[YC], errors="coerce")
ydplot = ydplot.dropna(subset=[TC,YC])
fig = px.scatter(ydplot, x=TC, y=YC, color="Alloy_name", labels={TC:"Temperature (°C)",YC:"Yield strength (MPa)"}, title="Recorded rows and selected-alloy model curve")
fig.add_scatter(x=temps, y=yc["predicted_mpa"], mode="lines", name=f"{alloy} model curve")
fig.add_vline(x=yield_temp, line_dash="dash", annotation_text="Selected temperature")
st.plotly_chart(fig, use_container_width=True)

at = np.linspace(float(pd.to_numeric(gd["aging_temperature_c"], errors="coerce").min()), float(pd.to_numeric(gd["aging_temperature_c"], errors="coerce").max()), 100)
gc = pd.DataFrame([{
    "aging_temperature_c": float(t),
    "log10_aging_time_hours": float(np.log10(aging_time)),
    "nickel_content_wt_percent": float(ni),
    "aluminum_content_wt_percent": float(al),
    "titanium_content_wt_percent": float(ti),
    "tantalum_content_wt_percent": float(ta),
} for t in at])[gamma_features]

def gamma_plot(title, model_key, current, y_label):
    fig = px.line(x=at, y=models[model_key].predict(gc), labels={"x":"Aging temperature (°C)","y":y_label}, title=title)
    fig.add_vline(x=aging_temp, line_dash="dash", annotation_text="Selected")
    fig.add_scatter(x=[aging_temp], y=[current], mode="markers", name="Current estimate")
    st.plotly_chart(fig, use_container_width=True)

c1,c2 = st.columns(2)
with c1: gamma_plot("γ′ particle radius vs aging temperature","radius",rhat,"Radius (nm)")
with c2: gamma_plot("γ′ volume fraction vs aging temperature","volume",vhat,"Volume fraction (%)")
gamma_plot("Interparticle spacing vs aging temperature","spacing",shat,"Spacing (nm)")

result = pd.DataFrame([{
    "alloy":alloy, "yield_temperature_c":yield_temp, "predicted_yield_strength_mpa":yhat,
    "nickel_wt_percent":ni, "aluminum_wt_percent":al, "titanium_wt_percent":ti, "tantalum_wt_percent":ta,
    "aging_temperature_c":aging_temp, "aging_time_hours":aging_time,
    "predicted_gamma_prime_radius_nm":rhat, "predicted_gamma_prime_volume_fraction_percent":vhat,
    "predicted_interparticle_spacing_nm":shat
}])
st.download_button("Download predictions as CSV", result.to_csv(index=False).encode("utf-8"), "combined_materials_prediction.csv", "text/csv")
with st.expander("Show detailed prediction table"):
    st.dataframe(result, use_container_width=True)
