from flask import Flask, render_template, request
import joblib
import numpy as np


# Load the trained model
model = joblib.load('model.pkl')

# Create the Flask application
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the inputs from the user
    area = float(request.form['area'])
    bathrooms = float(request.form['bathrooms'])
    bedrooms = float(request.form['bedrooms'])
    mainroad = float(request.form['mainroad'])
    basement = float(request.form['basement'])
    parking = float(request.form['parking'])
    furnishingstatus = float(request.form['furnishingstatus'])
    

    # ... preprocess the other inputs as needed ...

    # Make the prediction
    X = np.array([area, bathrooms, bedrooms,mainroad,basement,parking,furnishingstatus]).reshape(1, -1)
    prediction = model.predict(X)[0]

    # Render the template with the prediction
    return render_template('index.html', prediction=prediction)

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)