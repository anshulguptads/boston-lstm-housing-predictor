
# Boston Housing Price Predictor using LSTM

This repository provides an end-to-end solution to train, save, and deploy an LSTM model for predicting MEDV (Median value of owner-occupied homes) using the Boston Housing dataset.

---

## 📦 Project Structure

```
.
├── train_model.py         # Trains and saves the LSTM model
├── predict_api.py         # FastAPI app for real-time prediction
├── streamlit_app.py       # Streamlit frontend to interact with the model
├── boston.csv             # Dataset file
├── requirements.txt       # Python dependencies
└── README.md              # Instructions
```

---

## 🚀 How to Run

### 🔧 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 🧠 2. Train Model
```bash
python train_model.py
```

### ⚡ 3. Run FastAPI for Real-Time Inference
```bash
uvicorn predict_api:app --reload
```

### 🌐 4. Run Streamlit UI
```bash
streamlit run streamlit_app.py
```

---

## 📤 Deployment

Upload your repository to GitHub and deploy the `streamlit_app.py` on [Streamlit Cloud](https://streamlit.io/cloud). Make sure your `requirements.txt` is properly defined.

