from flask import Flask, request, jsonify
from flask_cors import CORS
from nhaTotPredict.predict_house_price import predict_house_price 
from nhaTotPredict.predict_apartment_price import predict_apartment_price 
from nhaTotPredict.predict_land_price import predict_land_price 
from nhaTotPredict.predict_commercial_price import predict_commercial_price 

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/housePredict": {"origins": "*"}})

@app.route('/housePredict', methods=['POST'])
def housePredict():
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
        urgent = data.get('urgent')
        pty_characteristics = data.get('pty_characteristics')

        # Call the predict_house_price function
        predicted_price = predict_house_price(ward, district, size, rooms, toilets, floors, house_type, furnishing_sell,urgent,pty_characteristics)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price:,.0f} VND"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/apartmentPredict', methods=['POST'])
def apartmentPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        ward = data.get('ward')
        district = data.get('district')
        size = data.get('size')
        rooms = data.get('rooms')
        toilets = data.get('toilets')
        apartment_type = data.get('apartment_type')
        furnishing_sell = data.get('furnishing_sell')
        urgent = data.get('urgent')

        # Call the predict_apartment_price function
        predicted_price = predict_apartment_price(ward, district, size, rooms, toilets, apartment_type, furnishing_sell,urgent)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price:,.0f} VND"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/landPredict', methods=['POST'])
def landPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        ward = data.get('ward')
        district = data.get('district')
        size = data.get('size')
        land_type = data.get('land_type')
        pty_characteristics = data.get('pty_characteristics')
        urgent = data.get('urgent')

        # Call the predict_land_price function
        predicted_price = predict_land_price(ward, district, size, land_type, pty_characteristics,urgent)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price:,.0f} VND"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/commercialPredict', methods=['POST'])
def commercialPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        ward = data.get('ward')
        district = data.get('district')
        size = data.get('size')
        commercial_type = data.get('commercial_type')
        furnishing_sell = data.get('furnishing_sell')
        urgent = data.get('urgent')

        # Call the predict_commercial__price function
        predicted_price = predict_commercial_price(ward, district, size, commercial_type, furnishing_sell,urgent)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price:,.0f} VND"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
