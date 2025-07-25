import os
import requests # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import tensorflow as tf # pyright: ignore[reportMissingImports]
import pickle
from flask import Flask, request, jsonify # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import nltk # pyright: ignore[reportMissingImports]
import warnings

# --- Helper Function to Download Models ---
def download_file_from_hf(repo_id, filename, local_dir="models"):
    """Downloads a file from a Hugging Face Hub repo."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    local_path = os.path.join(local_dir, filename)

    # Only download if the file doesn't already exist
    if not os.path.exists(local_path):
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded {filename} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {filename}. Error: {e}")
            return None
    return local_path
REPO_ID = "Bhargav1111111111/ai-text-analysis-dashboard" 

FN_MODEL_PATH = download_file_from_hf(REPO_ID, "fake_news_cnn_model.h5")
FN_TOKENIZER_PATH = download_file_from_hf(REPO_ID, "tokenizer.pkl")
MT_MODEL_PATH = download_file_from_hf(REPO_ID, "multi_task_model.h5")
MT_TOKENIZER_PATH = download_file_from_hf(REPO_ID, "tokenizer1.pkl")

app = Flask(__name__)

# --- Load Models and Tokenizers from downloaded files ---
try:
    fake_news_model = tf.keras.models.load_model(FN_MODEL_PATH)
    with open(FN_TOKENIZER_PATH, 'rb') as f:
        fake_news_tokenizer = pickle.load(f)
    print("✅ Fake News Detector model loaded.")

    multi_task_model = tf.keras.models.load_model(MT_MODEL_PATH)
    with open(MT_TOKENIZER_PATH, 'rb') as f:
        multi_task_tokenizer = pickle.load(f)
    print("✅ Multi-Task Classifier model loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")


# --- Define Constants ---
MAX_LEN_FAKE_NEWS = 200
MAX_LEN_MULTI_TASK = 50

# --- Label Mappings from Notebooks ---
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
violence_labels = ['Physical Violence', 'Sexual Violence', 'Emotional Violence', 'Economic Violence', 'Harmful Traditional Practice']
hate_labels = ['Hate Speech', 'Offensive Speech', 'Normal']

# --- API Endpoint for Fake News Detection ---
@app.route('/api/detect-fake-news', methods=['POST'])
def detect_fake_news():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the text
        sequences = fake_news_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=MAX_LEN_FAKE_NEWS)
        
        # Predict
        prediction_prob = fake_news_model.predict(padded)[0][0]
        prediction = int(prediction_prob > 0.5)

        result = {
            'prediction': 'Fake-News' if prediction == 1 else 'Real-News',
            'confidence': float(prediction_prob) if prediction == 1 else float(1 - prediction_prob)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API Endpoint for Multi-Task Classification ---
@app.route('/api/classify-text', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess the text
        sequences = multi_task_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=MAX_LEN_MULTI_TASK, padding='post')

        # Prepare input for the model's heads
        input_data = {
            'emotion_input': padded,
            'violence_input': padded,
            'hate_input': padded
        }
        
        # Predict
        preds = multi_task_model.predict(input_data, verbose=0)
        
        result = {
            "emotion": emotion_labels[np.argmax(preds[0])],
            "violence": violence_labels[np.argmax(preds[1])],
            "hate": hate_labels[np.argmax(preds[2])]
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

    app.run(debug=True, port=5000)
if __name__ == "__main__":
    app.run()
