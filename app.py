"""Flask web app: Heart Disease Risk Prediction with Explainable AI"""
import json, joblib, pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
B = joblib.load("best_model.pkl")
PIPE, FEATS, BG, LABELS = B["pipeline"], B["features"], B["background"], B["labels"]
SUMMARY = json.load(open("static/summary.json"))

FIELDS = [  # name, label, type, options/range
 ("age", "Age (years)", "num", (20, 100, 1)),
 ("sex", "Sex", "sel", {1: "Male", 0: "Female"}),
 ("cp", "Chest pain type", "sel", {0: "Asymptomatic", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"}),
 ("trestbps", "Resting blood pressure (mm Hg)", "num", (80, 220, 1)),
 ("chol", "Serum cholesterol (mg/dl)", "num", (100, 600, 1)),
 ("fbs", "Fasting blood sugar > 120 mg/dl", "sel", {0: "No", 1: "Yes"}),
 ("restecg", "Resting ECG result", "sel", {1: "Normal", 2: "ST-T abnormality", 0: "LV hypertrophy"}),
 ("thalach", "Maximum heart rate achieved", "num", (60, 220, 1)),
 ("exang", "Exercise-induced angina", "sel", {0: "No", 1: "Yes"}),
 ("oldpeak", "ST depression (oldpeak)", "num", (0, 7, 0.1)),
 ("slope", "Slope of peak exercise ST", "sel", {0: "Downsloping", 1: "Flat", 2: "Upsloping"}),
 ("ca", "Major vessels coloured by fluoroscopy", "sel", {0: "0", 1: "1", 2: "2", 3: "3"}),
 ("thal", "Thalassemia", "sel", {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}),
]
DEFAULTS = {"age": 54, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240, "fbs": 0, "restecg": 1,
            "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}

def explain(row):
    """Local XAI: replace each feature with the training median and measure how much the
    predicted probability changes. Positive = pushes towards disease."""
    base = PIPE.predict_proba(row[FEATS])[0, 1]
    out = []
    for f in FEATS:
        r2 = row.copy(); r2[f] = BG[f]
        delta = base - PIPE.predict_proba(r2[FEATS])[0, 1]
        out.append({"feature": LABELS[f], "value": row[f].iloc[0], "impact": round(float(delta) * 100, 1)})
    return sorted(out, key=lambda d: abs(d["impact"]), reverse=True)

@app.route("/", methods=["GET", "POST"])
def index():
    vals, result = dict(DEFAULTS), None
    if request.method == "POST":
        vals = {f[0]: float(request.form[f[0]]) for f in FIELDS}
        row = pd.DataFrame([vals])
        prob = float(PIPE.predict_proba(row[FEATS])[0, 1])
        result = {"prob": round(prob * 100, 1), "label": "High risk of heart disease" if prob >= .5 else "Low risk of heart disease",
                  "high": prob >= .5, "xai": explain(row)}
    return render_template("index.html", fields=FIELDS, vals=vals, result=result, s=SUMMARY)

if __name__ == "__main__":
    app.run(debug=True)
