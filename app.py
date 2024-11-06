from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/trainHousePredictModel', methods=['POST'])
def trainHousePredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('houseDataset.csv')
        X = df[['WardId', 'DistrictId', 'Size', 'Rooms', 'Toilets', 'Floors', 'Type', 'FurnishingSell', 'Urgent', 'Characteristics']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['Type', 'FurnishingSell', 'Characteristics', 'Urgent', 'WardId', 'DistrictId']
        numerical_features = ['Size', 'Rooms', 'Toilets', 'Floors']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'house_predict_model.pkl')

        # Return success message as JSON response
        return jsonify(dict(message='Model saved successfully!')), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/trainApartmentPredictModel', methods=['POST'])
def trainApartmentPredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('apartmentDataset.csv')
        X = df[['WardId', 'DistrictId', 'Size', 'Rooms', 'Toilets', 'Type', 'FurnishingSell', 'Urgent']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['Type', 'FurnishingSell', 'Urgent', 'WardId', 'DistrictId']
        numerical_features = ['Size', 'Rooms', 'Toilets']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),       # Chuẩn hóa các cột số
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-Hot Encoding cho các cột phân loại
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'apartment_predict_model.pkl')

        # Return the predicted price as a JSON response
        return jsonify({
            'message': 'Save model success!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/trainLandPredictModel', methods=['POST'])
def trainLandPredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('landDataset.csv')
        X = df[['WardId', 'DistrictId', 'Size', 'Type', 'Urgent', 'Characteristics']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['Type', 'Characteristics', 'Urgent', 'WardId', 'DistrictId']
        numerical_features = ['Size']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),       # Chuẩn hóa các cột số
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-Hot Encoding cho các cột phân loại
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'land_predict_model.pkl')

        # Return the predicted price as a JSON response
        return jsonify({
            'message': 'Save model success!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/trainCommercialPredictModel', methods=['POST'])
def trainCommercialPredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('commercialDataset.csv')
        X = df[['WardId', 'DistrictId', 'Size', 'Type', 'Urgent', 'FurnishingSell']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['Type', 'FurnishingSell', 'Urgent', 'WardId', 'DistrictId']
        numerical_features = ['Size']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),       # Chuẩn hóa các cột số
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-Hot Encoding cho các cột phân loại
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'commercial_predict_model.pkl')

        # Return the predicted price as a JSON response
        return jsonify({
            'message': 'Save model success!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/housePredict', methods=['POST'])
def housePredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        wardId = data.get('wardId')
        districtId = data.get('districtId')
        size = data.get('size')
        rooms = data.get('rooms')
        toilets = data.get('toilets')
        floors = data.get('floors')
        type = data.get('type')
        furnishingSell = data.get('furnishingSell')
        urgent = data.get('urgent')
        characteristics = data.get('characteristics')

        new_house = pd.DataFrame({
            'WardId': [wardId],
            'DistrictId': [districtId],
            'Size': [size],
            'Rooms': [rooms],
            'Toilets': [toilets],
            'Floors': [floors],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent],
            'Characteristics': [characteristics],
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('house_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_house)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price[0]:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/apartmentPredict', methods=['POST'])
def apartmentPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        wardId = data.get('wardId')
        districtId = data.get('districtId')
        size = data.get('size')
        rooms = data.get('rooms')
        toilets = data.get('toilets')
        type = data.get('type')
        furnishingSell = data.get('furnishingSell')
        urgent = data.get('urgent')

        new_apartment = pd.DataFrame({
            'WardId': [wardId],
            'DistrictId': [districtId],
            'Size': [size],
            'Rooms': [rooms],
            'Toilets': [toilets],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent],  
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('apartment_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_apartment)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price[0]:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/landPredict', methods=['POST'])
def landPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        wardId = data.get('wardId')
        districtId = data.get('districtId')
        size = data.get('size')
        type = data.get('type')
        characteristics = data.get('characteristics')
        urgent = data.get('urgent')

        new_land = pd.DataFrame({
            'WardId': [wardId],
            'DistrictId': [districtId],
            'Size': [size],
            'Type': [type],
            'Urgent': [urgent],  
            'Characteristics': [characteristics],
        })

        # 8. Tải lại mô hình đã lưu
        loaded_model = joblib.load('land_predict_model.pkl')

        # 9. Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_land)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price[0]:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/commercialPredict', methods=['POST'])
def commercialPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        wardId = data.get('wardId')
        districtId = data.get('districtId')
        size = data.get('size')
        type = data.get('type')
        furnishingSell = data.get('furnishingSell')
        urgent = data.get('urgent')

        new_commercial = pd.DataFrame({
            'WardId': [wardId],
            'DistrictId': [districtId],
            'Size': [size],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent],  
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('commercial_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_commercial)

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{predicted_price[0]:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
