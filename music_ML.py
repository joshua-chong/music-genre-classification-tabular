# music_ML.py
import io
import os
import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Your feature extractor (as given)
# -----------------------------
def extract_features(y, sr):
    features = {}

    # length
    features['length'] = len(y)

    # chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = float(np.mean(chroma_stft))
    features['chroma_stft_var'] = float(np.var(chroma_stft))

    # rms
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = float(np.mean(rms))
    features['rms_var'] = float(np.var(rms))

    # spectral_centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
    features['spectral_centroid_var'] = float(np.var(spec_centroid))

    # spectral_bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
    features['spectral_bandwidth_var'] = float(np.var(spec_bw))

    # rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = float(np.mean(rolloff))
    features['rolloff_var'] = float(np.var(rolloff))

    # zero_crossing_rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate_mean'] = float(np.mean(zcr))
    features['zero_crossing_rate_var'] = float(np.var(zcr))

    # harmony (tonnetz has 6 dimensions → take mean & var of each)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features['harmony_mean'] = float(np.mean(tonnetz))
    features['harmony_var'] = float(np.var(tonnetz))

    # perceptr (onset strength, 1D array)
    perceptr = librosa.onset.onset_strength(y=y, sr=sr)
    features['perceptr_mean'] = float(np.mean(perceptr))
    features['perceptr_var'] = float(np.var(perceptr))

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # tempo can be numpy scalar; cast safely
    features['tempo'] = float(tempo)

    # mfccs (1–20)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = float(np.mean(mfccs[i-1]))
        features[f'mfcc{i}_var'] = float(np.var(mfccs[i-1]))

    return features


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Music Genre/Tag Predictor", page_icon="🎵", layout="centered")

st.title("🎵 Music Predictor")
st.caption("Upload a .wav file → extract features → run model prediction")


# -----------------------------
# Load artifacts (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts(path="genre_rf.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Put model.pkl next to this script, or update the path."
        )
    bundle = joblib.load(path)
    try:
        model, feature_names, target_names = bundle
    except Exception:
        # Fall back if saved differently; adapt as needed
        model = bundle
        feature_names = None
        target_names = None
    return model, feature_names, target_names


# -----------------------------
# Utility: prepare feature vector in correct order
# -----------------------------
def build_feature_frame(features_dict, feature_names=None):
    """Return a 1-row DataFrame aligned to feature_names. Missing features -> 0, extra features dropped."""
    df = pd.DataFrame([features_dict])

    if feature_names is None:
        # If the model was trained without exposing feature_names, we just use whatever order we have.
        # (Not ideal; better to persist feature_names during training!)
        return df

    # Ensure all expected columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    # Keep only expected columns, in order
    df = df[feature_names]
    return df


# -----------------------------
# Sidebar: model path (optional)
# -----------------------------
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="genre_rf.pkl", help="Path to your joblib-saved model bundle")
    target_sr = st.number_input("Target sample rate (Hz)", value=22050, step=100, help="Audio will be loaded at this rate")
    duration_cap = st.number_input("Optional duration cap (seconds)", value=0, step=1, help="0 = no cap. If >0, trims audio.")

# Load model once user interacts (and show helpful error inline)
load_err = None
try:
    model, feature_names, target_names = load_artifacts(model_path)
except Exception as e:
    load_err = str(e)


# -----------------------------
# Main: uploader and prediction
# -----------------------------
file = st.file_uploader("Upload a WAV file", type=["wav"], accept_multiple_files=False)

if load_err:
    st.error(load_err)

if file is not None and not load_err:
    st.subheader("Preview")
    st.audio(file)

    # Read bytes once, then pass a fresh buffer to librosa (it consumes the stream)
    raw_bytes = file.read()

    try:
        # librosa.load supports file-like objects; ensure we pass a new BytesIO
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=target_sr, mono=True)

        if duration_cap and duration_cap > 0:
            y = y[: int(duration_cap * sr)]

        st.write(f"Loaded audio: {len(y)} samples @ {sr} Hz")

        with st.spinner("Extracting features..."):
            feats = extract_features(y, sr)
            X = build_feature_frame(feats, feature_names)

        st.success("Features extracted")
        with st.expander("See extracted features"):
            st.dataframe(X.T.rename(columns={0: "value"}))

        with st.spinner("Running prediction..."):
            # Support classifiers with predict_proba; if not available, just predict
            y_pred = model.predict(X)[0]
            st.subheader("Prediction")
            if target_names is not None:
                pred_label = target_names[y_pred] if isinstance(y_pred, (int, np.integer)) and y_pred < len(target_names) else str(y_pred)
            else:
                pred_label = str(y_pred)
            st.markdown(f"**Predicted class:** `{pred_label}`")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                # Build a probability table with class names if we have them
                if target_names is not None and len(target_names) == len(probs):
                    prob_df = pd.DataFrame({"class": target_names, "probability": probs}).sort_values("probability", ascending=False)
                else:
                    prob_df = pd.DataFrame({"class": range(len(probs)), "probability": probs}).sort_values("probability", ascending=False)

                st.subheader("Class probabilities")
                st.dataframe(prob_df.reset_index(drop=True))

    except Exception as e:
        st.exception(e)


# -----------------------------
# Helpful notes
# -----------------------------
with st.expander("ℹ️ Tips / Troubleshooting"):
    st.markdown(
        """
- Ensure **model.pkl** contains `(model, feature_names, target_names)`.  
  - Example when saving after training:
    ```python
    joblib.dump((model, feature_names, target_names), "model.pkl")
    ```
- If you trained without `feature_names`, set it to the exact column list used during training and re-save.
- If you see `FileNotFoundError: model.pkl`, put the file next to this script or set the correct path in the sidebar.
- If tempo or other features error on very short clips, try uploading a longer segment or lowering the duration cap to skip noisy tails.
        """
    )
