import os
import pickle
import requests
import streamlit as st

# --- Download models from Hugging Face ---
def download_file_from_hf(repo_id, filename, local_dir="models"):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        st.info(f"Downloading {filename} from Hugging Face...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"✅ Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to download {filename}: {e}")
            return None
    return local_path

# --- Load model ---
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- Constants ---
REPO_ID = "Bhargav1111111111/ai-text-analysis-dashboard"
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
violence_labels = ['Physical Violence', 'Sexual Violence', 'Emotional Violence', 'Economic Violence', 'Harmful Traditional Practice']
hate_labels = ['Hate Speech', 'Offensive Speech', 'Normal']

# --- Download and Load All Models ---
with st.spinner("🔁 Loading models..."):
    fn_model_path = download_file_from_hf(REPO_ID, "fake_news_model.pkl")
    emo_model_path = download_file_from_hf(REPO_ID, "emotion_model.pkl")
    vio_model_path = download_file_from_hf(REPO_ID, "violence_model.pkl")
    hate_model_path = download_file_from_hf(REPO_ID, "hate_model.pkl")

    fake_news_model = load_model(fn_model_path)
    emotion_model = load_model(emo_model_path)
    violence_model = load_model(vio_model_path)
    hate_model = load_model(hate_model_path)

st.title("🧠 AI Text Analysis Dashboard (Streamlit)")
st.caption("Detect Fake News or Classify text by Emotion, Violence, and Hate Speech using Scikit-learn models.")

# --- Interface ---
option = st.radio("Choose Task", ["Fake News Detection", "Multi-Task Classification"])

text = st.text_area("📝 Enter your text here:", height=200)

if st.button("Analyze"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("🔍 Analyzing..."):
            if option == "Fake News Detection":
                prob = fake_news_model.predict_proba([text])[0][1]
                is_fake = prob > 0.5
                st.success("Prediction: " + ("Fake-News" if is_fake else "Real-News"))
                st.info(f"Confidence: {prob * 100:.2f}%")
            else:
                emotion = emotion_labels[emotion_model.predict([text])[0]]
                violence = violence_labels[violence_model.predict([text])[0]]
                hate = hate_labels[hate_model.predict([text])[0]]

                st.subheader("🔎 Classification Results")
                st.markdown(f"**Emotion:** {emotion}")
                st.markdown(f"**Violence Type:** {violence}")
                st.markdown(f"**Hate Speech Level:** {hate}")
