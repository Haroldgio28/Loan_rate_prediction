
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb


# Load the model

input_file = 'model1.0.bin'


# Load the model from the pickle file
with open(input_file, 'rb') as f_in: 
    dv,  model = pickle.load(f_in)


app = Flask('rate')

@app.route('/predict', methods=['POST'])

def predict():
    cust = request.get_json()
    X = dv.transform([cust])
    X = xgb.DMatrix(X)
    y_pred = model.predict(X)[0]
    rate = round(y_pred, 2)
    result = {
        'rate': float(rate)
    }
    return jsonify(result)
# jsonify(round(result['rate'],2))

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0',port=9696)


