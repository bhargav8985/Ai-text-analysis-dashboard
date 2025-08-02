import os
import pickle
import requests
from flask import Flask, request, jsonify

# Flask App Initialization
app = Flask(__name__)

# Download models from Hugging Face
def download_file_from_hf(repo_id, filename, local_dir="models"):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {filename}: {e}")
            return None
    return local_path

# Hugging Face repo ID
REPO_ID = "Bhargav1111111111/ai-text-analysis-dashboard"

# File names
FN_MODEL_PATH = download_file_from_hf(REPO_ID, "fake_news_model.pkl")
EMOTION_MODEL_PATH = download_file_from_hf(REPO_ID, "emotion_model.pkl")
VIOLENCE_MODEL_PATH = download_file_from_hf(REPO_ID, "violence_model.pkl")
HATE_MODEL_PATH = download_file_from_hf(REPO_ID, "hate_model.pkl")

# Load models
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    fake_news_model = load_model(FN_MODEL_PATH)
    emotion_model = load_model(EMOTION_MODEL_PATH)
    violence_model = load_model(VIOLENCE_MODEL_PATH)
    hate_model = load_model(HATE_MODEL_PATH)
    print("✅ All Scikit-learn models loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Label mappings
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
violence_labels = ['Physical Violence', 'Sexual Violence', 'Emotional Violence', 'Economic Violence', 'Harmful Traditional Practice']
hate_labels = ['Hate Speech', 'Offensive Speech', 'Normal']

# --- API: Fake News Detection ---
@app.route('/api/detect-fake-news', methods=['POST'])
def detect_fake_news():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        prob = fake_news_model.predict_proba([text])[0][1]
        prediction = int(prob > 0.5)

        result = {
            'prediction': 'Fake-News' if prediction == 1 else 'Real-News',
            'confidence': float(prob) if prediction == 1 else float(1 - prob)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API: Multi-Task Classification ---
@app.route('/api/classify-text', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = {
            "emotion": emotion_labels[emotion_model.predict([text])[0]],
            "violence": violence_labels[violence_model.predict([text])[0]],
            "hate": hate_labels[hate_model.predict([text])[0]]
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
