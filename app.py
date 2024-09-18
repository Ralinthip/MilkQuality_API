from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("milknew.pkl")

@app.route('/api/milk', methods=['POST'])
def milk():
    pH = float(request.form.get('pH')) 
    temprature = int(request.form.get('temprature')) 
    taste = int(request.form.get('taste')) 
    odor = int(request.form.get('odor')) 
    fat = int(request.form.get('fat')) 
    turbidity = int(request.form.get('turbidity')) 
    
    # Prepare the input for the model
    x = np.array([[pH, temprature, taste, odor, fat, turbidity]])

    # Predict using the model
    prediction = model.predict(x)

    # Return the result
    if int(prediction[0]) == 0:
        return {'Milk Quality': 'Low'}, 200   
    elif int(prediction[0]) == 1:
        return {'Milk Quality': 'Medium'}, 200   
    elif int(prediction[0]) == 2:
        return {'Milk Quality': 'High'}, 200   

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

