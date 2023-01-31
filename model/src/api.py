from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from pandas import DataFrame
from pickle import load
from os import path
from model import trainAndSaveModel
from data_modification import formatData
from sys import exit

app = Flask(__name__, template_folder=path.join("..", "templates"))
cors = CORS(app)

single_predict_route = "/single_predict"
multi_predict_route = "/multi_predict"

@app.route("/", methods=["GET"])
def start():
    return render_template("index.html")

@app.route(single_predict_route, methods=["POST"])
@cross_origin()
def single_predict():
    json_data = request.get_json()
    data = DataFrame(json_data, index=[0])
    data = formatData(data)
    data = scaler.transform(data)
    prediction = model.predict(data)
    result = {
        "solar_radiation": prediction[0]
    }
    return jsonify(result)

@app.route(multi_predict_route, methods=["POST"])
@cross_origin()
def multi_predict():
    json_data = request.get_json()
    data = DataFrame(json_data) 
    data = formatData(data)
    data = scaler.transform(data)
    predictions = model.predict(data)
    result = { 
        "solar_radiations": [
            {
                "solar_radiation": prediction
            }
            for prediction in predictions 
        ]
    }
    return jsonify(result) 

if(__name__ == "__main__"):
    if(not path.exists(path.join("..", "resources", "solar_radiation_prediction_model.pkl"))):
        trainAndSaveModel()
    if(not path.exists(path.join("..", "resources", "scaler.pkl"))):
        print("Scaler not found.")
        exit()
    global model
    model = load(file=open(path.join("..", "resources", "solar_radiation_prediction_model.pkl"), "rb"))
    global scaler
    scaler = load(file=open(path.join("..", "resources", "scaler.pkl"), "rb"))
    app.run(debug=True)