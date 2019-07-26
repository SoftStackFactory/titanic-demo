from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify({"what":'Hello World...'})



@app.route('/predict', methods = ['GET', 'POST'])
def process_data():

# Grab JSON input
    input_data = request.get_json()
    

# Initiate Variables
    age = input_data['age']
    gender = input_data['gender']
    was_alone = input_data['was_alone']

    model = joblib.load('./titanic_grad_boost.joblib')


# mock data
    mock_data = {
        'age': 38,
        'was_alone': 1,
        'gender': 1
    }

# process
    mock_input = [mock_data['age'], mock_data['was_alone'], mock_data['gender']]
    input_data = [age, was_alone, gender]
    input_data = np.array(input_data).reshape(1, -1)
    # print(input_data)

# predict
    pred = model.predict(input_data)[0]
    prob_of_survival = model.predict_proba(input_data)[0][1]

# result
    if pred == 1:
        result = 'survived'
    else:
        result = 'perished'

<<<<<<< HEAD
    # result = 'survived' if pred == 1 else 'perished'

=======
>>>>>>> f7157b7abc1a83998ee3bdb1635c68414c50ac34
    chance_of_survival= round(prob_of_survival, 2)


    return jsonify({"result":result, "chance":chance_of_survival})


if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True, port=5000) #run app in debug mode on port 5000
