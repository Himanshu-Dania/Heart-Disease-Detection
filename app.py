from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the models
cb1 = pickle.load(open('cbheart1.pkl', 'rb'))
cb2 = pickle.load(open('cbheart2.pkl', 'rb'))
lr = pickle.load(open('LRHD_0.8F1_0.75Acc.pkl', 'rb'))
gb = pickle.load(open('GradBoostHeartDisease_0.82_0.82.pkl', 'rb'))

# Define the ensemble weights
weights = np.array([0.25014892, 0.24990093, 0.24985107, 0.25009908])

app = Flask(__name__)
def validate_input(data):
    # Convert input data to the format expected by the models
    data['Sex'] = 1 if data['Sex'].lower() == 'male' else 0
    chest_pain_types = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']
    for i, chest_pain_type in enumerate(chest_pain_types):
        data['ChestPainType_' + str(i)] = data['ChestPainType'].lower() == chest_pain_type
    del data['ChestPainType']
    data['FastingBS'] = 1 if data['FastingBS'] > 120 else 0
    data['ExerciseAngina'] = 1 if data['ExerciseAngina'].lower() == 'yes' else 0

    # Convert the data to a flat list of features
    features = [data['Age'], data['Sex'], data['RestingBP'], data['Cholesterol'], data['MaxHR'], data['ExerciseAngina']] + [data['ChestPainType_' + str(i)] for i in range(4)]

    # Define the expected data types for each feature
    expected_types = [int, int, int, int, int, int, bool, bool, bool, bool]
    constraints = [(20,80), (0,1), (0,200), (0,603), (60,202), (0,1), (False,True), (False,True), (False,True), (False,True)]
    
    # Check if the number of features is correct
    if len(features) != len(expected_types):
        return False, "Incorrect number of features. Expected {} but got {}.".format(len(expected_types), len(features)), None
    
    # Check the data type and constraints of each feature
    for i in range(len(features)):
        if type(features[i]) != expected_types[i]:
            return False, "Incorrect data type for feature {}. Expected {} but got {}.".format(i, expected_types[i].__name__, type(features[i]).__name__), None
        if features[i] < constraints[i][0] or features[i] > constraints[i][1]:
            return False, "Feature {} out of bounds. Expected between {} and {} but got {}.".format(i, constraints[i][0], constraints[i][1], features[i]), None
    # If all checks pass, return True and the features
    return True, "Input is valid.", features

        
@app.route('/',methods=["GET"])
def home():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    is_valid, message, features = validate_input(data)
    if not is_valid:
        print(message)
        return jsonify({'error': message}),400
    # Make prediction using the models
    prediction1 = cb1.predict_proba([np.array(features)])
    prediction2 = cb2.predict_proba([np.array(features)])
    prediction3 = lr.predict_proba([np.array(features)])
    prediction4 = gb.predict_proba([np.array(features)])

    # Compute the ensemble prediction
    ensemble_prediction = np.argmax(np.average(np.array([prediction1, prediction2, prediction3, prediction4]), axis=0, weights=weights))

    # Send back to the client
    output = {'prediction': int(ensemble_prediction)}
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
