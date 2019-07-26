from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pickle
import numpy as np

# Create Flask App
app = Flask(__name__)


# Home Route
@app.route('/')
def hello_world():
    return jsonify({"what":'Hello World...'})


# Prediction Route
@app.route('/predict', methods = ['GET', 'POST'])
def process_data():

# Grab JSON input
    input_data = request.get_json()
    
# Initiate Variables
    age = input_data['age']
    gender = input_data['gender']
    was_alone = input_data['was_alone']

# Load Model
    model = joblib.load('./titanic_grad_boost.joblib')

# process data as model input
    input_data = [age, was_alone, gender]
    input_data = np.array(input_data).reshape(1, -1)

# Make prediction 
    pred = model.predict(input_data)[0]
    prob_of_survival = model.predict_proba(input_data)[0][1]

# Process results
    result = 'survived' if pred == 1 else 'perished'
    chance_of_survival= round(prob_of_survival, 2)

# Return Payload
    return jsonify({"result":result, "chance":chance_of_survival})


# Run app 
if __name__ == '__main__':
    app.run(host= '0.0.0.0') #run app in debug mode on port 5000