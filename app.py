import shap
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import sqlite3
from datetime import datetime
import google.generativeai as genai


app = Flask(__name__)

# ==========================================
# GEMINI CONFIGURATION
# ==========================================
genai.configure(api_key="AIzaSyCBv-N1ximef9rrwJkSc9WtIyzbptcsZ90")



# ==========================================
# LOAD XGBOOST MODEL
# ==========================================
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ==========================================
# SQLITE DATABASE
# ==========================================
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            tenure REAL,
            monthly_charges REAL,
            contract TEXT,
            internet_service TEXT,
            risk_probability REAL,
            churn_prediction TEXT,
            retention_strategy TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()

# ==========================================
# HOME PAGE
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')


# ==========================================
# CHATBOT PAGE
# ==========================================
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


# ==========================================
# PREDICTION ENGINE
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():

    try:

        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])

        contract = request.form['contract']
        internet_service = request.form['internet_service']
        payment_method = request.form['payment_method']

        gender = request.form['gender']
        senior_citizen = int(request.form['senior_citizen'])
        partner = request.form['partner']
        dependents = request.form['dependents']
        phone_service = request.form['phone_service']
        multiple_lines = request.form['multiple_lines']

        online_security = request.form['online_security']
        online_backup = request.form['online_backup']
        device_protection = request.form['device_protection']
        tech_support = request.form['tech_support']
        streaming_tv = request.form['streaming_tv']
        streaming_movies = request.form['streaming_movies']

        paperless_billing = request.form['paperless_billing']

        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
       }])

        input_encoded = pd.get_dummies(input_data)

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        probability = model.predict_proba(input_encoded)[0][1]


        # SHAP Explainable AI

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(input_encoded)

        feature_impacts = []

        for feature, value in zip(
            input_encoded.columns,
            shap_values[0]
        ):
            feature_impacts.append(
                (feature, float(value))
          )

        feature_impacts.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = feature_impacts[:5]

        shap_explanations = []

        for feature, impact in top_features:

            if feature == "MonthlyCharges":
                text = "High monthly charges are increasing churn risk."

            elif feature == "Contract_Month-to-month":
                text = "Month-to-month contracts typically have higher churn rates."

            elif feature == "tenure":
                text = "Customer tenure is significantly influencing the prediction."

            elif feature == "OnlineSecurity_No":
                text = "Online security service is affecting customer retention."

            elif feature == "TotalCharges":
                text = "Customer billing history contributes to churn risk."

            else:
                text = feature

            shap_explanations.append((text, impact))



        risk_percentage = float(round(probability * 100, 2))
        print("Probability:", probability)
        print("Risk Percentage:", risk_percentage)
        print("Type:", type(risk_percentage))

        status = (
            "High Risk"
            if risk_percentage > 50
            else "Low Risk"
        )

        # ==================================
        # GEMINI RETENTION STRATEGY
        # ==================================
        try:

            ai_model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
            You are a telecom customer retention expert.

            Customer Details:
            - Tenure: {tenure} months
            - Monthly Charges: ${monthly_charges}
            - Contract Type: {contract}
            - Internet Service: {internet_service}
            - Churn Risk: {risk_percentage}%

            Provide:
            1. A personalized retention strategy.
            2. A recommended offer or incentive.
            3. A short explanation.

            Keep the response professional and concise.
            """

            
            
            response = ai_model.generate_content(prompt)

            print(response.text)

            strategy = response.text

        except Exception as e:

            print("\n========== GEMINI ERROR ==========")
            print(e)
            print("==================================\n")

            strategy = "Fallback Strategy"
            

        # ==================================
        # SAVE TO DATABASE
        # ==================================
        conn = get_db_connection()

        conn.execute("""
            INSERT INTO logs
            (
                timestamp,
                tenure,
                monthly_charges,
                contract,
                internet_service,
                risk_probability,
                churn_prediction,
                retention_strategy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tenure,
            monthly_charges,
            contract,
            internet_service,
            risk_percentage,
            status,
            strategy

        ))

        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            probability=risk_percentage,
            status=status,
            strategy=strategy,
            top_features=shap_explanations
                    
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# ==========================================
# DASHBOARD
# ==========================================
@app.route('/dashboard')
def dashboard():

    conn = get_db_connection()

    logs = conn.execute(
        "SELECT * FROM logs ORDER BY timestamp DESC"
    ).fetchall()

    total_records = len(logs)

    total_churned = sum(
        1 for log in logs
        if log['churn_prediction'] == 'High Risk'
    )

    total_safe = total_records - total_churned

    churn_rate = (
        round((total_churned / total_records) * 100, 2)
        if total_records > 0
        else 0
    )

    

    # ==================================
    # PIE CHART
    # ==================================
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["High Risk", "Low Risk"],
                values=[total_churned, total_safe],
                hole=0.4
            )
        ]
    )

    fig.update_layout(
        title="Customer Churn Risk Distribution"
    )

    chart_html = fig.to_html(full_html=False)

    # ==================================
    # FEATURE IMPORTANCE CHART
    # ==================================
    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(10)
    )

    feature_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Most Important Churn Factors"
    )

    feature_fig.update_layout(
        height=500
    )

    feature_chart_html = feature_fig.to_html(
        full_html=False
    )

    # ==========================
    # CHURN TREND CHART
    # ==========================

    df_logs = pd.DataFrame([dict(log) for log in logs])

    trend_chart_html = ""

    if not df_logs.empty:

        df_logs["date"] = pd.to_datetime(
        df_logs["timestamp"]
        ).dt.date

        daily_churn = (
        df_logs[df_logs["churn_prediction"] == "High Risk"]
        .groupby("date")
        .size()
        .reset_index(name="count")
        )

        trend_fig = px.line(
        daily_churn,
        x="date",
        y="count",
        markers=True,
        title="Daily High-Risk Customer Trend"
        )

        trend_chart_html = trend_fig.to_html(
        full_html=False
        )


    conn.close()

    return render_template(
        "dashboard.html",
        db_logs=logs,
        total_records=total_records,
        total_churned=total_churned,
        churn_rate=churn_rate,
        chart_html=chart_html,
        feature_chart_html=feature_chart_html,
        trend_chart_html=trend_chart_html
    )


# ==========================================
# CHATBOT API
# ==========================================
@app.route('/chat_response', methods=['POST'])
def chat_response():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if user_message == "":
            return jsonify({
                "response": "Please enter a message."
            })

        try:
            ai_model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
                You are an AI Telecom Customer Retention Assistant.

                You have knowledge of these customer features:

                - Gender
                - SeniorCitizen
                - Partner
                - Dependents
                - Tenure
                - PhoneService
                - MultipleLines
                - InternetService
                - OnlineSecurity
                - OnlineBackup
                - DeviceProtection
                - TechSupport
                - StreamingTV
                - StreamingMovies
                - Contract
                - PaperlessBilling
                - PaymentMethod
                - MonthlyCharges
                - TotalCharges

                Answer only based on these telecom churn concepts.
                Do not ask for network logs, complaints, or usage history.

                User Question:
                {user_message}

                Provide concise business-oriented answers.
                """

            response = ai_model.generate_content(prompt)

            return jsonify({
                "response": response.text
            })

        except Exception as e:
            print("GEMINI ERROR:", str(e))

            return jsonify({
                "response": f"Gemini Error: {str(e)}"
            })

    except Exception as e:
        return jsonify({
            "response": f"System Error: {str(e)}"
        })


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)