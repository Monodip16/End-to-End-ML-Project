from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## Route for home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict_data():
    if request.method == "GET":
        return render_template("prediction.html")
    else:
        data = CustomData(
            gender = request.form.get("gender"),
            race_ethnicity = request.form.get("race_ethnicity"),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            lunch = request.form.get("lunch"),
            test_preparation_course = request.form.get("test_preparation_course"),
            reading_score = int(request.form.get("reading_score")),
            writing_score = int(request.form.get("writing_score"))
        )

        data_df = data.get_data_as_dataframe()
        print(data_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_df)
        return render_template("prediction.html", results = results[0]) 


if __name__ == "__main__":
    app.run(host = "localhost", port = 8080, debug = True)
