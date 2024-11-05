import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_and_save_model():
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
    joblib.dump(pipeline, 'house_price_model.pkl')

def predict_house_price(wardId, districtId, size, rooms, toilets, floors, type, furnishingSell, urgent, characteristics):
    # Create a DataFrame for the new house
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
    loaded_model = joblib.load('house_price_model.pkl')

    # Dự đoán giá cho ngôi nhà mới
    predicted_price = loaded_model.predict(new_house)

    return predicted_price[0]
