# importing the necessary dependencies
from flask import Flask, render_template, request, jsonify
import sklearn
import pickle
import numpy as np
import joblib
import pandas as pd
app = Flask(__name__)  # initializing a flask app


@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def index():
    # # reading the inputs given by the user
    Triceps_skinfold = (request.form['Triceps_skinfold'])
    Diabetes_fun = (request.form['Diabetes_function'])
    pregnant = (request.form['pregnant'])
    Plasma_glucose = (request.form['Plasma_glucose'])
    Diastolic_bp = (request.form['Diastolic_bp'])
    Hour_serum = (request.form['Hour_serum'])
    bmi = (request.form['bmi'])
    Age = (request.form['Age'])

    co = ['Number of times pregnant', 'Plasma glucose concentration',
          'Diastolic blood pressure (mm Hg)', 'Triceps skinfold thickness (mm)',
          '2-Hour serum insulin (mu U/ml)',
          'Body mass index (weight in kg/(height in m)^2)',
          'Diabetes pedigree function', 'Age']

    int_features = [x for x in request.form.values()]

    final_features = np.array([[Triceps_skinfold,Diabetes_fun,pregnant,Plasma_glucose,Diastolic_bp,Hour_serum,bmi,Age]])

    filename = "xgboost_model_suraj.pickle"
    gg= "scaler_model.pickle"
    scaler_model = pickle.load(open(gg, 'rb'))
    d = scaler_model.transform(final_features)
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    # predictions using the loaded model file
    prediction = loaded_model.predict(d)

    print('prediction is', prediction[0])
    # showing the prediction results in a UI
    return render_template('results.html', prediction=prediction[0])






if __name__ == "__main__":
    app.run(debug=True)  # running the app
