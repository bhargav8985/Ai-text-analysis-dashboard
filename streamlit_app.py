import os
import requests
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Helper to Download File from Hugging Face ---
def download_file_from_hf(repo_id, filename, local_dir="models"):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        st.info(f"📥 Downloading {filename} ...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"✅ Downloaded {filename}")
        except Exception as e:
            st.error(f"❌ Failed to download {filename}: {e}")
            return None
    return local_path

# --- Constants ---
REPO_ID = "Bhargav1111111111/ai-text-analysis-dashboard"
MAX_LEN_FAKE_NEWS = 200
MAX_LEN_MULTI_TASK = 50

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
violence_labels = ['Physical Violence', 'Sexual Violence', 'Emotional Violence', 'Economic Violence', 'Harmful Traditional Practice']
hate_labels = ['Hate Speech', 'Offensive Speech', 'Normal']

# --- Load models and tokenizers ---
@st.cache_resource
def load_models():
    fn_model = tf.keras.models.load_model(download_file_from_hf(REPO_ID, "fake_news_cnn_model.h5"))
    mt_model = tf.keras.models.load_model(download_file_from_hf(REPO_ID, "multi_task_model.h5"))

    with open(download_file_from_hf(REPO_ID, "tokenizer.pkl"), "rb") as f:
        fn_tokenizer = pickle.load(f)

    with open(download_file_from_hf(REPO_ID, "tokenizer1.pkl"), "rb") as f:
        mt_tokenizer = pickle.load(f)

    return fn_model, fn_tokenizer, mt_model, mt_tokenizer

# Load everything
with st.spinner("🔁 Loading models and tokenizers..."):
    fake_news_model, fake_news_tokenizer, multi_task_model, multi_task_tokenizer = load_models()

# --- Streamlit UI ---
st.title("🧠 AI Text Analysis Dashboard")
st.caption("Detect Fake News or classify text by Emotion, Violence, and Hate using TensorFlow models.")

task = st.radio("Select Task", ["Fake News Detection", "Multi-Task Classification"])
text = st.text_area("📝 Enter your text below", height=200)

if st.button("Analyze"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("🔍 Analyzing..."):
            if task == "Fake News Detection":
                seq = fake_news_tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=MAX_LEN_FAKE_NEWS)
                prob = fake_news_model.predict(padded)[0][0]
                st.success(f"Prediction: {'Fake-News' if prob > 0.5 else 'Real-News'}")
                st.info(f"Confidence: {prob * 100:.2f}%")
            else:
                seq = multi_task_tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=MAX_LEN_MULTI_TASK, padding='post')
                input_data = {
                    'emotion_input': padded,
                    'violence_input': padded,
                    'hate_input': padded
                }
                preds = multi_task_model.predict(input_data, verbose=0)
                st.subheader("🔎 Classification Results")
                st.markdown(f"**Emotion:** {emotion_labels[np.argmax(preds[0])]}")
                st.markdown(f"**Violence Type:** {violence_labels[np.argmax(preds[1])]}")
                st.markdown(f"**Hate Speech Level:** {hate_labels[np.argmax(preds[2])]}")
