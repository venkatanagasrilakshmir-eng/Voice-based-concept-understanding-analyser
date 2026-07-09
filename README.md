# 🎙️ Voice-Based Concept Understanding Analyser (VBCUA)

> An AI-powered application that evaluates a user's conceptual understanding using speech recognition, semantic analysis, and machine learning.

## 📖 Table of Contents

- Overview
- Features
- Technologies
- Project Structure
- Installation
- Usage
- Docker
- Testing
- Workflow
- Future Enhancements
- Author
- License

---

## 📌 Overview

VBCUA analyzes a user's spoken explanation of a concept by converting speech to text, extracting audio features, comparing semantic similarity with a reference answer, generating scores, and creating a PDF report.

---

## ✨ Features

- 🎤 Audio Upload
- 📝 Speech-to-Text (Whisper)
- 🧠 Semantic Analysis (SBERT)
- 🔊 Audio Feature Extraction
- 📊 Performance Scoring
- 📄 PDF Report Generation
- 💾 Database Storage
- 🌐 Streamlit Interface
- ⚡ FastAPI Backend
- 🐳 Docker Support

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Frontend |
| FastAPI | Backend API |
| Whisper | Speech Recognition |
| Sentence-BERT | Semantic Similarity |
| Librosa | Audio Processing |
| SQLite | Database |
| Docker | Containerization |
| Pytest | Testing |

---

## 📂 Project Structure

```text
vbcua/
├── .github/
│   └── workflows/
├── api/
├── backend/
├── frontend/
├── uploads/
├── reports/
├── data/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── README.md
└── LICENSE
```

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/vbcua.git
cd vbcua

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Run Streamlit

```bash
streamlit run frontend/app.py
```

---

## ▶️ Run FastAPI

```bash
uvicorn api.main:app --reload
```

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## ✅ Run Tests

```bash
pytest
```

---

## 🔄 Workflow

```text
User
   │
   ▼
Upload Audio
   │
   ▼
Speech-to-Text
   │
   ▼
Audio Feature Extraction
   │
   ▼
Semantic Analysis
   │
   ▼
Scoring Engine
   │
   ▼
Generate PDF Report
   │
   ▼
Store in Database
   │
   ▼
Display Results
```

---

## 📈 Future Enhancements

- Real-time Voice Analysis
- Multi-language Support
- AI Feedback Suggestions
- Cloud Deployment
- Dashboard Analytics

---

## 👩‍💻 Author

**R. Venkata Naga Srilakshmi**

B.Tech – Artificial Intelligence & Data Science

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you like this project, don't forget to star the repository.# Voice-based-concept-understanding-analyser
