from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pyodbc

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Cấu hình kết nối SQL Server
def get_db_connection():
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                          'SERVER=localhost;'
                          'DATABASE=BatDongSan;'
                          'UID=sa;'
                          'PWD=123456789')
    return conn

# API lấy danh sách dữ liệu từ bảng 'real_estates'
@app.route('/getRealEstates', methods=['POST'])
def get_real_estates():
    # Get query parameters from the request
    data = request.get_json()

    category_id = int(data.get('categoryId')) if data.get('categoryId') is not None else None
    price = int(data.get('price')) if data.get('price') is not None else None
    ward_id = int(data.get('wardId')) if data.get('wardId') is not None else None
    size = float(data.get('size')) if data.get('size') is not None else None
    rooms = int(data.get('rooms')) if data.get('rooms') is not None else None
    toilets = int(data.get('toilets')) if data.get('toilets') is not None else None
    floors = int(data.get('floors')) if data.get('floors') is not None else None
    estate_type = data.get('type') if data.get('type') is not None else None
    furnishing_sell = data.get('furnishingSell') if data.get('furnishingSell') is not None else None
    urgent = data.get('urgent') if data.get('urgent') is not None else None
    characteristics = data.get('characteristics') if data.get('characteristics') is not None else None

    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Construct base SQL query
    query = 'SELECT * FROM HCMRealEstate WHERE Status = ? AND CategoryId = ? AND WardId = ?  AND DeleteStatus = ?'
    params = ['Đang hiển thị', category_id, ward_id, 'FALSE']

    # Append filters based on the parameters
    if rooms and rooms > 0:
        query += ' AND Rooms = ?'
        params.append(rooms)

    if toilets and toilets > 0:
        query += ' AND Toilets = ?'
        params.append(toilets)

    if floors and floors > 0:
        query += ' AND Floors = ?'
        params.append(floors)

    if estate_type:
        query += ' AND Type = ?'
        params.append(estate_type)

    if furnishing_sell:
        query += ' AND FurnishingSell = ?'
        params.append(furnishing_sell)

    if characteristics:
        query += ' AND Characteristics = ?'
        params.append(characteristics)

    if urgent:
        query += ' AND Urgent = ?'
        if (urgent == "0"):
            params.append(False)
        else:
            params.append(True)

    if price:
        min_price = price - 1000000000
        max_price = price + 1000000000
        query += ' AND (Price >= ? AND Price <= ?)'
        params.extend([min_price, max_price])

    if size:
        min_size = size - 20
        max_size = size + 20
        query += ' AND (Size >= ? AND Size <= ?)'
        params.extend([min_size, max_size])

    # Execute the query with the parameters
    cursor.execute(query, params)
    real_estates = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Format the results as a list of dictionaries
    real_estates_list = [
        {
            'RealEstateId': estate[0],
            'CategoryId': estate[1],
            'WardId': estate[2],
            'UserId': estate[3],
            'Address': estate[4],
            'Title': estate[5],
            'Description': estate[6],
            'TypePost': estate[7],
            'Size': estate[8],
            'Price': estate[9],
            'Unit': estate[10],
            'Direction': estate[11],
            'BalconyDirection': estate[12],
            'FurnishingSell': estate[13],
            'Rooms': estate[14],
            'Toilets': estate[15],
            'Floors': estate[16],
            'Type': estate[17],
            'PropertyStatus': estate[18],
            'PropertyLegalDocument': estate[19],
            'Characteristics': estate[20],
            'Urgent': estate[21],
            'Images': estate[22],
        } for estate in real_estates
    ]

    # Return results as JSON
    return jsonify(real_estates_list)

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
        urgent_numeric = int(urgent) if urgent is not None else 0
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
            'Urgent': [urgent_numeric],
            'Characteristics': [characteristics],
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('house_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_house)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
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
        urgent_numeric = int(urgent) if urgent is not None else 0

        new_apartment = pd.DataFrame({
            'WardId': [wardId],
            'DistrictId': [districtId],
            'Size': [size],
            'Rooms': [rooms],
            'Toilets': [toilets],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent_numeric],  
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('apartment_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_apartment)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
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
        urgent_numeric = int(urgent) if urgent is not None else 0

        new_land = pd.DataFrame({
            'WardId': [wardId],
            'DistrictId': [districtId],
            'Size': [size],
            'Type': [type],
            'Urgent': [urgent_numeric],  
            'Characteristics': [characteristics],
        })

        # 8. Tải lại mô hình đã lưu
        loaded_model = joblib.load('land_predict_model.pkl')

        # 9. Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_land)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
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
        urgent_numeric = int(urgent) if urgent is not None else 0

        new_commercial = pd.DataFrame({
            'WardId': [wardId],
            'DistrictId': [districtId],
            'Size': [size],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent_numeric],  
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('commercial_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_commercial)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
