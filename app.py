from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load('model_pipeline.pkl')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Log the received data for debugging
    print("Received data:", data)

    input_data = pd.DataFrame([data])

    # Check if input data matches expected columns
    expected_columns = [
        'precipitation(mm/day)', 'drainage(mm/day)', 'Elevation(m)', 
        'Water Table(coastal region)', 'urbanization', 'runoff coefficient'
    ]
    
    if not all(col in input_data.columns for col in expected_columns):
        return jsonify({
            'status': 'error',
            'message': 'Input data does not match expected format'
        }), 400  # Bad Request

    try:
        prediction = model.predict(input_data)[0]
        response = {
            'waterlogging_prediction': prediction,
            'status': 'success'
        }
    except Exception as e:
        response = {
            'status': 'error',
            'message': str(e)
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
