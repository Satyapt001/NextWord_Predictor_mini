from pathlib import Path

from flask import Flask, jsonify, render_template, request

from predictor import NextWordPredictor


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


app = Flask(__name__)


predictor = NextWordPredictor(
    MODEL_DIR
)


@app.route("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():

    data = request.get_json(
        silent=True
    ) or {}

    text = data.get(
        "text",
        ""
    ).strip()

    if not text:
        return jsonify({
            "suggestions": []
        })


    suggestions = predictor.predict(
        text,
        top_k=5
    )


    return jsonify({
        "suggestions": suggestions
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )