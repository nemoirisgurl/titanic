from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)
model = joblib.load("models/titanicModel.joblib")
model_columns = joblib.load("models/model_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        data = request.form.to_dict()
        data["Age"] = int(data["Age"])
        data["SibSp"] = int(data["SibSp"])
        data["Parch"] = int(data["Parch"])
        data["Fare"] = float(data["Fare"])
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
        if data["Sex"] == "female":
            data["Title"] = "Mrs" if data["Age"] > 18 else "Miss"
        else:
            data["Title"] = "Mr" if data["Age"] > 18 else "Master"
        print("Received form data:", data)
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(df)[0]
        result = "Survived" if prediction == 1 else "Did not survive"
        return f"<h2>Prediction: {result}</h2><a href='/'>Go back</a>"
    except ValueError as e:
        return f"<h2>Error: {str(e)}</h2>"


@app.route("/graph")
def graph():
    data = pd.read_csv("train.csv")
    fig = px.histogram(data, x="Age", color="Survived", barmode="group")
    graph = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return render_template("graph.html", graph=graph)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.get_dummies(pd.DataFrame(data))
        df = df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(df)
        return jsonify({"Survived": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=3000, debug=True)
