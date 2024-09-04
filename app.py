from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__)

CORS(app)

# Load your prepared DataFrame
# Assuming df is already prepared as in your code
# You can load it from a file or from memory if running the script directly

df = pd.read_csv('dataset.csv')

# Assuming df is already prepared
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['district', 'province']])

# Combine encoded features with numeric features
X = np.hstack((encoded_features.toarray(), df[['area', 'bedroom', 'toilet']].values))
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract the input features from the request
    district = data['district']
    province = data['province']
    area = data['area']
    bedroom = data['bedroom']
    toilet = data['toilet']

    # Create a DataFrame for the new house
    new_house = pd.DataFrame({
        'district': [district],
        'province': [province],
        'area': [area],
        'bedroom': [bedroom],
        'toilet': [toilet]
    })

    # Encode the new house data
    encoded_new_house = encoder.transform(new_house[['district', 'province']]).toarray()
    new_house_features = np.hstack((encoded_new_house, new_house[['area', 'bedroom', 'toilet']].values))

    # Predict the price
    predicted_price = model.predict(new_house_features)

    # Return the predicted price as a JSON response
    return jsonify({
        'predicted_price': f"{predicted_price[0]:,.0f} VND"
    })
    # return jsonify(predicted_price=predicted_price[0]), 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    app.run(debug=True)
