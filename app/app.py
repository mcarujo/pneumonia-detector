import argparse
import logging

from flask import Flask, jsonify, redirect, render_template, request

from model import ModelPredict, ModelTrain

app = Flask(__name__)

logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@app.route("/")
@app.route("/index")
def index():
    logging.info("Route index.")
    return render_template("index.html")


@app.route("/dashboard")
@app.route("/dashboard/<lines>")
def dashboard(lines=100):
    log_file = open("logs.txt", "r")
    file_lines = log_file.readlines()
    file_lines.reverse()
    last_lines = file_lines[:lines]
    log_file.close()
    return render_template("dashboard.html", log_lines=last_lines)


@app.route("/train", methods=["GET", "POST"])
def train():
    """
    Train process starting by a request.
    """

    logging.info("Route train.")
    # Loading the dataset tools
    logging.info("Loading the dataset tools.")
    logging.info(
        "Creating the ModelTrain class passing the dataset as argument.")
    training = ModelTrain()
    metrics = training.run()

    metrics = [
        ("Accuracy", metrics["ac"]),
        ("F1 Score", metrics["f1"]),
        ("Log Loss", metrics["ll"]),
        ("Precision", metrics["ps"]),
        ("Recall", metrics["rc"]),
    ]
    # Returning results as HTML
    logging.info("Returning results as HTML.")
    if request.method == "GET":
        return render_template("train.html", metrics=metrics)
    else:
        # Returning results as json
        return jsonify(metrics)


@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image_path = "temp/" + image.filename
            image.save(image_path)
            return redirect("/predict?image_path=" + image.filename)
    return render_template("form.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Prediction the next days.
    """

    model = ModelPredict()
    image_path = request.args["image_path"]
    prediction = model.predict(image_path)
    if prediction:
        label = model.define_label(prediction)
        logging.info(f"Image : {image_path} - Prediction : {prediction}")
        return render_template(
            "predict.html",
            accuracy=prediction,
            class_=label,
            filename=image_path,
        )
    else:
        return render_template(
            "no_prediction.html",
        )


if __name__ == "__main__":

    # parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())
    if args["debug"]:
        app.run(debug=True, host="0.0.0.0", port=5080)
    else:
        app.run(host="0.0.0.0", threaded=True, port=5080)
