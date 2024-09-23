from flask import Flask, request, jsonify
from flask_cors import CORS
from nhaTotPredict.predict import predict_house_price  # Import prediction function

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/housePredict": {"origins": "*"}})

@app.route('/housePredict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        ward = data.get('ward')
        district = data.get('district')
        size = data.get('size')
        rooms = data.get('rooms')
        toilets = data.get('toilets')
        floors = data.get('floors')
        house_type = data.get('house_type')
        furnishing_sell = data.get('furnishing_sell')

        # Call the predict_house_price function
        predicted_price = predict_house_price(ward, district, size, rooms, toilets, floors, house_type, furnishing_sell)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price:,.0f} VND"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
