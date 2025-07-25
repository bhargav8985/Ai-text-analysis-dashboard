
# 🧠 AI Text Analysis Dashboard

A full-stack deep learning application for detecting fake news and classifying online text into emotion, hate speech, and violence categories. Combines NLP techniques and TensorFlow models served via a Flask API with a responsive Tailwind CSS frontend.

---

## 🚀 Live Demo
🔗 [GitHub Repository](https://github.com/bhargav8985/Ai-text-analysis-dashboard)

---

## 💡 Features

- 🔍 **Fake News Detector**: Classifies news articles as Real or Fake using a CNN-based model.
- 🎭 **Multi-Task Classifier**: Detects:
  - **Emotion** (e.g., joy, sadness, anger)
  - **Hate Speech** (e.g., offensive, normal)
  - **Violence** (e.g., emotional, physical)
- 🧰 **RESTful APIs**: Flask backend serving real-time model predictions.
- 🌐 **Interactive Frontend**: Built with Tailwind CSS and vanilla JS.
- ☁️ **Hugging Face Integration**: Downloads pretrained models from Hugging Face Hub.

---

## 🧪 Tech Stack

**Frontend:**
- HTML
- Tailwind CSS
- JavaScript

**Backend:**
- Python
- Flask
- TensorFlow

**Machine Learning:**
- CNN, LSTM
- Tokenization
- TF-IDF
- Multi-label classification

**Deployment:**
- Flask (API server)
- Hugging Face Hub (model hosting)
- Local or cloud-based frontend (e.g., Vercel)

---

## 🧰 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Bhargav8985/Ai-text-analysis-dashboard.git
cd ai-text-analysis-dashboard
```

### 2. Install Dependencies

#### Option A: Using pip and virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
conda create --name ai-text-env python=3.10
conda activate ai-text-env
pip install -r requirements.txt
```

The required packages include:

```ini
Flask==2.3.2
flask-cors==4.0.0
gunicorn==21.2.0
tensorflow==2.11.0
pandas==2.2.2
scikit-learn==1.4.2
numpy==1.26.4
```

### 3. Run the Backend API

```bash
python app.py
```

The Flask server will start at [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 4. Open the Frontend

Open `index.html` in your browser (either double-click or use a live server extension in VSCode).

---

## 🧠 Model Info

- **Fake News Model**: CNN trained on Kaggle news dataset.
- **Multi-Task Model**: Multi-head model classifying:
  - Emotion (6 classes)
  - Violence Type (5 classes)
  - Hate Speech (3 classes)

Model files and tokenizers are downloaded automatically from:  
🔗 [Hugging Face Model Repo](https://huggingface.co/Bhargav1111111111/ai-text-analysis-dashboard)

---

## 📌 Notes

- The Fake News Detector is optimized for American English news.
- Frontend includes simulated results in case backend API is unavailable.
- All models are dynamically downloaded and loaded at runtime.

---

---

## 📝 License

This project is licensed under the **MIT License**.  
Feel free to use, fork, or contribute!


