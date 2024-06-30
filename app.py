from flask import Flask, render_template, request
import pandas as pd
import pickle

import sklearn

app = Flask(__name__)
car = pd.read_csv('cleaned_car.csv')

# Load the model
model = pickle.load(open('randomforestreg.pkl', 'rb'))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())
    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    car_model = request.form['car_model']
    year = int(request.form['year'])
    fuel_type = request.form['fuel_type']
    kilo_driven = int(request.form['kilo_driven'])

    # Create a dataframe with the input data
    input_data = pd.DataFrame({
        'company': [company],
        'name': [car_model],
        'year': [year],
        'fuel_type': [fuel_type],
        'kms_driven': [kilo_driven]
    })

    # Predict the price
    prediction = model.predict(input_data)[0]

    return render_template(
        'index.html',
        prediction_text=f'The predicted price of the car is {prediction:.2f}',
        companies=sorted(car['company'].unique()),
        car_models=sorted(car['name'].unique()),
        years=sorted(car['year'].unique(), reverse=True),
        fuel_types=sorted(car['fuel_type'].unique())
    )

if __name__ == '__main__':
    app.run(debug=True)
