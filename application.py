from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd

# Load the model and dataset
MODEL_PATH = 'D:/Download folder/Car_price_prediction/final.pkl'
DATASET_PATH = 'D:/Download folder/Car_price_prediction/car_new_final_o.xls'

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

dataset = pd.read_csv(DATASET_PATH)

# Extract dropdown options
car_names = sorted(dataset['name'].unique())
companies = sorted(dataset['company'].unique())
fuel_types = sorted(dataset['fuel_type'].unique())

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/car_prediction')
def car_prediction():
    return render_template('index.html', car_names=car_names, companies=companies, fuel_types=fuel_types)

'''@app.route('/laptop_prediction')
def laptop_prediction():
    return render_template('laptop_prediction.html')

@app.route('/house_prediction')
def house_prediction():
    return render_template('house_prediction.html')'''

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form['name']
    company = request.form['company']
    fuel_type = request.form['fuel_type']
    kms_driven = int(request.form['kms_driven'])
    year = int(request.form['year'])

    # Prepare the input for prediction
    input_data = pd.DataFrame({
        'name': [name],
        'company': [company],
        'fuel_type': [fuel_type],
        'kms_driven': [kms_driven],
        'year': [year]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    return render_template('index.html',
                           prediction=f"Estimated Price: ₹{int(prediction):,}",
                           car_names=car_names,
                           companies=companies,
                           fuel_types=fuel_types)

if __name__ == '__main__':
    app.run(debug=True)
