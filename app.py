import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Models and Tokenizers ---
try:
    # Load Fake News Detection Model
    fake_news_model = tf.keras.models.load_model('fake_news_cnn_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        fake_news_tokenizer = pickle.load(f)
    print("✅ Fake News Detector model and tokenizer loaded successfully.")

    # Load Multi-Task Classification Model
    multi_task_model = tf.keras.models.load_model('multi_task_model.h5')
    with open('tokenizer1.pkl', 'rb') as f:
        multi_task_tokenizer = pickle.load(f)
    print("✅ Multi-Task Classifier model and tokenizer loaded successfully.")

except Exception as e:
    print(f"❌ Error loading models or tokenizers: {e}")

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