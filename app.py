from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model_data = joblib.load("saved_models/rf_all_model.pkl")
model = model_data["model"]

scaler = joblib.load("saved_models/scaler.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")


@app.route("/", methods=["GET","POST"])
def home():

    result = ""
    confidence = ""
    top_factors = []
    alternatives = []

    if request.method == "POST":

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])


        # Feature Engineering
        NPK_sum = N + P + K
        NP_ratio = N / (P + 1)
        NK_ratio = N / (K + 1)
        PK_ratio = P / (K + 1)

        N_P_interaction = N * P
        N_K_interaction = N * K
        P_K_interaction = P * K

        temp_humidity_interaction = temperature * humidity
        rainfall_humidity_ratio = rainfall / (humidity + 1)
        temp_rainfall_interaction = temperature * rainfall

        soil_health_score = (
            (ph / 7.0) * 0.3 +
            (N / 140) * 0.25 +
            (P / 145) * 0.25 +
            (K / 205) * 0.2
        )

        climate_index = (
            (temperature / 43.68) * 0.4 +
            (humidity / 100) * 0.3 +
            (rainfall / 298.56) * 0.3
        )

        avg_npk = (N + P + K) / 3

        nutrient_balance = 1 - (
            (abs(N - avg_npk) +
             abs(P - avg_npk) +
             abs(K - avg_npk)) / (3 * avg_npk + 1)
        )

        data = [[
            N, P, K, temperature, humidity, ph, rainfall,
            NPK_sum, NP_ratio, NK_ratio, PK_ratio,
            N_P_interaction, N_K_interaction, P_K_interaction,
            temp_humidity_interaction,
            rainfall_humidity_ratio,
            temp_rainfall_interaction,
            soil_health_score,
            climate_index,
            nutrient_balance
        ]]

        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0]

        result = label_encoder.inverse_transform([prediction])[0]
        confidence = round(max(proba)*100,2)

        # Top alternative crops
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3_crops = label_encoder.inverse_transform(top3_idx)

        alternatives = []
        for i in range(1,3):
            alternatives.append({
                "crop": top3_crops[i],
                "prob": round(proba[top3_idx[i]]*100,2)
            })

        # Example explanation factors
        top_factors = [
            {"feature":"Humidity","impact":"+0.097"},
            {"feature":"Potassium","impact":"-0.077"},
            {"feature":"Rainfall/Humidity Ratio","impact":"+0.075"},
            {"feature":"Rainfall","impact":"+0.072"},
            {"feature":"Phosphorus","impact":"+0.047"}
        ]

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        top_factors=top_factors,
        alternatives=alternatives
    )


if __name__ == "__main__":
    app.run(debug=True)