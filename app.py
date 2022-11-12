from flask import Flask, render_template, request
from pycaret.classification import *

app = Flask(__name__)

model = load_model("models/final_model")
columns = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_values = [i for i in request.form.values()]
    final_form_values = np.array(form_values)
    data_unseen = pd.DataFrame([final_form_values], columns=columns)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.Label[0])
    result = ""
    if prediction == 1:
        result = "Воду можна вживати"
    else:
        result = "Воду не можна вживати"
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=False)
