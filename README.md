# ❤️ CardioSense – Heart Disease Prediction Web Application with Explainable AI (SDG 3)

CardioSense is a Machine Learning-based web application that predicts heart disease risk using 13 clinical attributes from the UCI Cleveland Heart Disease Dataset. It uses multiple classification algorithms and Explainable AI (XAI) to provide transparent predictions and visualize the factors influencing each result.

## 🚀 Features

- Heart disease prediction (High Risk / Low Risk).
- Analysis of 13 clinical attributes.
- Implementation of 5 Machine Learning algorithms.
- Explainable AI (XAI) for individual predictions.
- Global feature importance visualization.
- Model performance comparison and evaluation.
- Interactive web interface using Flask.
- Supports early heart disease risk screening.

## 📂 Project Structure

```text
CardioSense/
│
├── heart.csv
├── train.py
├── app.py
├── requirements.txt
├── best_model.pkl
│
├── templates/
│   └── index.html
│
└── static/
    ├── charts/
    ├── results.csv
    └── summary.json
```

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- HTML & CSS

## 🤖 Machine Learning Algorithms

1. Decision Tree
2. Gaussian Naive Bayes
3. Logistic Regression
4. Support Vector Machine (SVM)
5. Random Forest

## 📊 Dataset

**UCI Cleveland Heart Disease Dataset**

- 303 patient records.
- 13 input features.
- 1 target variable.

The dataset contains clinical attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina and other medical measurements.

## ⚙️ Installation & Execution

**Step 1: Clone the repository**

```bash
git clone <your-github-repository-url>
```

**Step 2: Navigate to the project folder**

```bash
cd CardioSense
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Train the Machine Learning models**

```bash
python train.py
```

This performs data preprocessing, feature selection, model training, evaluation and generates charts and result files. It also saves the best model.

**Step 5: Run the Flask application**

```bash
python app.py
```

**Step 6: Open the application**

Visit the following URL in your browser:

```text
http://127.0.0.1:5000
```

## 📈 Model Performance

The best-performing model was Logistic Regression using all 13 features.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 86.9% |
| Precision | 88.5% |
| Recall | 82.1% |
| F1-Score | 85.2% |
| ROC-AUC | 93.2% |
| 5-Fold CV Accuracy | 85.5% |

## 💡 Explainable AI (XAI)

CardioSense provides explanations for individual predictions and global feature importance.

- Identifies important clinical factors influencing predictions.
- Uses visual charts to show factors increasing or decreasing predicted risk.
- Helps users understand how the model reaches its predictions.

## 🌍 Sustainable Development Goal

**SDG 3 – Good Health and Well-being**

This project supports early heart disease risk screening and promotes accessible healthcare decision support.

## 🔮 Future Enhancements

- Integration with larger and local hospital datasets.
- SHAP-based explanations.
- Mobile application development.
- Secure patient database integration.
- Integration with hospital information systems.

## ⚠️ Disclaimer

CardioSense is an educational and research project intended for preliminary risk screening and decision support. It is not a substitute for professional medical advice, diagnosis or treatment.

## 👩‍💻 Author

**Padarthi Dhana Lakshmi**

B.Tech – Computer Science and Engineering

Lakireddy Bali Reddy College of Engineering

---

⭐ If you find this project interesting, consider giving the repository a star!