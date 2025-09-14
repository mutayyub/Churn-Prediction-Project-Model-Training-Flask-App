from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# load model, scaler, encoders
with open("finalized_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Total Spend", "Last Interaction"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, probability = None, None

    if request.method == "POST":
        data = {
            "Age": int(request.form["Age"]),
            "Tenure": int(request.form["Tenure"]),
            "Usage Frequency": int(request.form["Usage_Frequency"]),
            "Support Calls": int(request.form["Support_Calls"]),
            "Total Spend": float(request.form["Total_Spend"]),
            "Last Interaction": int(request.form["Last_Interaction"]),
            "Gender": request.form["Gender"],
            "Subscription Type": request.form["Subscription_Type"],
            "Contract Length": request.form["Contract_Length"]
        }

        df_input = pd.DataFrame([data])

        # encode categoricals
        for col, le in label_encoders.items():
            df_input[col] = le.transform(df_input[col].astype(str))

        # scale numerics
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        pred = model.predict(df_input.values)[0]
        prob = model.predict_proba(df_input.values)[0][1]

        prediction = "⚠️ Likely to Churn" if pred == 1 else "✅ Likely to Stay"
        probability = round(prob, 2)

    return render_template("index.html", prediction=prediction, probability=probability, request=request)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
