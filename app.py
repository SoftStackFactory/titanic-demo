from flask import Flask, request, jsonify
from sklearn.externals import joblib
import sys 
import pickle
import numpy as np
print('hello world')

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World...'



@app.route('/predict', methods = ['POST'])
def process_data():
    
    input_data = request.get_json()
    print(input_data)
    age = input_data['age']
    gender = input_data['gender']
    is_alone = input_data['is_alone']

    model = joblib.load('/titanic_grad_boost.joblib')


# mock data
    mock_data = {
        'age': 14,
        'was_alone': 0,
        'gender': 1
    }

# process
    mock_input = [mock_data['age'], mock_data['was_alone'], mock_data['gender']]
    # input_data = [input_data['age'], input_data['was_alone'], input_data['gender']]
    mock_input = np.array(input_data).reshape(1, -1)
    # print(input_data)

# predict
    pred = model.predict(mock_input)[0]
    prob_of_survival = model.predict_proba(mock_input)[0][1]

# result
    if pred == 1:
        result = 'survived'
    else:
        result = 'perished'
    
    chance_of_survival= round(prob_of_survival, 2)




    return jsonify({"result":result, "chance":chance_of_survival})


if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000
